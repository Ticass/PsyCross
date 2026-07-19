#if defined(PSYX_RENDERER_VULKAN)

#define SDL_MAIN_HANDLED
#include "PsyX/PsyX_public.h"
#include "PsyX/PsyX_render.h"
#include "PsyX/PsyX_globals.h"
#include "../gpu/PsyX_GPU.h"
#include "../platform.h"

#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "psx_vert_spv.h"
#include "psx_frag_spv.h"
#include "psx_rt_frag_spv.h"
#include "post_vert_spv.h"
#include "post_frag_spv.h"
#include "shadow_vert_spv.h"
#include "overlay_vert_spv.h"
#include "overlay_frag_spv.h"
#include "line_vert_spv.h"
#include "line_frag_spv.h"

extern SDL_Window* g_window;
extern "C" float PGXP_GetSzMax(void);
extern float g_PgxpFarWClamp;
void GR_InitPostProcess();

extern "C" {
float g_PsxPixelAspect = 1.0f;
float g_PsxWorldVScale = 0.872f;
float g_PsxWorldVShift = 20.0f;
float g_PsxWorldHScale = 1.0f;
int g_PsxFixedCamActive = 0;
int g_PsxCutsceneActive = 0;
int g_PsxUIOrthoPass = 0;
}

int g_windowWidth = 0, g_windowHeight = 0;
int g_dbg_wireframeMode = 0, g_dbg_texturelessMode = 0;
int g_PcHorPlusEnabled = 1, g_PcMenuPillarbox = 1, g_PcWidescreenMode = 1;
int g_cfg_pgxpTextureCorrection = 1, g_cfg_pgxpZBuffer = 1;
int g_PsxUsePgxp = 0, g_cfg_bilinearFiltering = 0, g_cfg_menuFilter = 0, g_cfg_affineTextures = 0;
int g_cfg_psxDither = 1, g_PsxDitherSuppressed = 0, g_cfg_msaaSamples = 0;
int g_cfg_rtgi = 0;
int g_cfg_postProcess = 0, g_cfg_tonemap = 0;
int g_PsyX_UsePerPixelFlashlight = 0, g_PsyX_FlashlightStyle = 0;
int g_PsyX_UseFlashlightShadows = 0, g_PsyX_FlashlightActive = 0;
int g_PsyX_ShadowsAllowed = 0, g_PsyX_FlashlightFpsMode = 0;
float g_PsyX_FlashlightShadowBias = 0.0018f;
float g_PsyX_FlashlightShadowNormalOffset = 0.0f;
float g_PsyX_FlashlightShadowStrength = 1.0f, g_PsyX_FlashlightShadowFadeDist = 0.0f;
float g_PsyX_FlashlightShadowFpsDrop = 0.0f;
float g_PsyX_FlashlightPos[3] = {}, g_PsyX_FlashlightShadowPos[3] = {};
float g_PsyX_FlashlightDir[3] = {0,0,1}, g_PsyX_FlashlightColor[3] = {1,1,1};
float g_PsyX_FlashlightInnerCos = 0.94f, g_PsyX_FlashlightOuterCos = 0.76f;
float g_PsyX_FlashlightRange = 4000.0f, g_PsyX_FlashlightSize = 3.0f;
float g_PsyX_FlashlightIntensity = 1.2f, g_PsyX_FlashlightSizeFps = 1.3f;
float g_PsyX_FlashlightIntensityFps = 2.1f;
float g_cfg_postProcessIntensity = 1.0f, g_cfg_tonemapIntensity = 1.0f;
float g_PsyX_FogColor[3] = {}, g_PsyX_FogStrength = 1.1f;
int g_PsxFogToBlack = 0, g_PsxSkipFramebufferStore = 0, g_PsxPresentLastFrame = 0;
int g_PsyX_ForceItemDepth = 0;
TextureID g_whiteTexture = 0, g_vramTexture = 0;
unsigned short vram[VRAM_WIDTH * VRAM_HEIGHT];

namespace {
constexpr uint32_t kMaxTextures = 16384;
constexpr VkDeviceSize kVertexBytes = sizeof(GrVertex) * MAX_VERTEX_BUFFER_SIZE;
constexpr uint32_t kUniformSlots = 8192;
constexpr uint32_t kFramebufferRegions = 8;
constexpr uint32_t kShadowMapSize = 1024;
constexpr uint32_t kMaxRayTriangles = 131072;
constexpr VkDeviceSize kOverlayUploadBytes = 8u * 1024u * 1024u;
constexpr VkDeviceSize kOverlayLineBytes = 1024u * 1024u;

struct Uniforms {
    float projection[16], projection3D[16];
    float fogColorStrength[4];
    float textureInfo[4];
    float hiresInfo[4];
    float renderInfo[4];
    float lightPosRange[4];
    float lightDirStyle[4];
    float lightColor[4];
    float effectInfo[4];
    float shadowMatrix[16];
    float shadowParams[4];
    float shadowClip[4];
    float framebufferRects[kFramebufferRegions][4];
};

struct PostConstants {
    int mode, tonemapMode;
    float texelSize[2];
    float time, postIntensity, tonemapIntensity, padding;
};

struct OverlayConstants { float rect[4], uv[4]; };

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDescriptorSet descriptor = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    bool alive = false;
    bool linear = false;
};

struct PipelineKey {
    uint8_t blend, depth, always, stencil, wire, offscreen, rayTracing;
    bool operator==(const PipelineKey& o) const { return blend==o.blend&&depth==o.depth&&always==o.always&&stencil==o.stencil&&wire==o.wire&&offscreen==o.offscreen&&rayTracing==o.rayTracing; }
};
struct PipelineEntry { PipelineKey key{}; VkPipeline pipeline = VK_NULL_HANDLE; };
struct VertexSlice { VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;void* mapped=nullptr;uint32_t vertexCount=0; };

/* std430-compatible material payload indexed by the BLAS primitive number.
 * This lets a ray-query hit reconstruct the PS1 texture/palette lookup rather
 * than returning a generic environment colour. */
struct RayMaterial {
    float uv01[4];
    float uv2Format[4];
    float pageClut[4];
    float color0[4];
    float color1[4];
    float color2[4];
};

struct RayScene {
    VkBuffer geometryBuffer=VK_NULL_HANDLE,materialBuffer=VK_NULL_HANDLE,blasBuffer=VK_NULL_HANDLE,blasScratch=VK_NULL_HANDLE;
    VkBuffer instanceBuffer=VK_NULL_HANDLE,tlasBuffer=VK_NULL_HANDLE,tlasScratch=VK_NULL_HANDLE;
    VkDeviceMemory geometryMemory=VK_NULL_HANDLE,materialMemory=VK_NULL_HANDLE,blasMemory=VK_NULL_HANDLE,blasScratchMemory=VK_NULL_HANDLE;
    VkDeviceMemory instanceMemory=VK_NULL_HANDLE,tlasMemory=VK_NULL_HANDLE,tlasScratchMemory=VK_NULL_HANDLE;
    VkAccelerationStructureKHR blas=VK_NULL_HANDLE,tlas=VK_NULL_HANDLE;
    uint32_t triangleCount=0;
};

struct VulkanState {
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = UINT32_MAX;
    VkQueue queue = VK_NULL_HANDLE;
    PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugLabel = nullptr;
    PFN_vkGetBufferDeviceAddressKHR getBufferDeviceAddress = nullptr;
    PFN_vkCreateAccelerationStructureKHR createAccelerationStructure = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroyAccelerationStructure = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR getAccelerationStructureBuildSizes = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAccelerationStructures = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getAccelerationStructureDeviceAddress = nullptr;
    bool rayQueryAvailable = false;
    char rayStatus[160] = "not initialized";
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkBuffer rtProbeBuffer = VK_NULL_HANDLE;
    VkDeviceMemory rtProbeMemory = VK_NULL_HANDLE;
    VkDeviceAddress rtProbeAddress = 0;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkFramebuffer> framebuffers;
    VkFormat depthFormat = VK_FORMAT_D24_UNORM_S8_UINT;
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;
    VkImage msaaColorImage = VK_NULL_HANDLE;
    VkDeviceMemory msaaColorMemory = VK_NULL_HANDLE;
    VkImageView msaaColorView = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkRenderPass loadRenderPass = VK_NULL_HANDLE, offscreenRenderPass = VK_NULL_HANDLE;
    VkFramebuffer offscreenFramebuffer = VK_NULL_HANDLE;
    VkImage offscreenDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory offscreenDepthMemory = VK_NULL_HANDLE;
    VkImageView offscreenDepthView = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE, complete = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint32_t imageIndex = 0;
    bool recording = false, recreate = false;
    bool fillModeNonSolid = false;
    int swapInterval = 1;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout rtSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout postSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout rtPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout postPipelineLayout = VK_NULL_HANDLE;
    VkShaderModule vert = VK_NULL_HANDLE, frag = VK_NULL_HANDLE;
    VkShaderModule rtFrag = VK_NULL_HANDLE;
    VkShaderModule postVert = VK_NULL_HANDLE, postFrag = VK_NULL_HANDLE;
    VkShaderModule overlayVert = VK_NULL_HANDLE, overlayFrag = VK_NULL_HANDLE;
    VkShaderModule lineVert = VK_NULL_HANDLE, lineFrag = VK_NULL_HANDLE;
    VkPipeline postPipeline = VK_NULL_HANDLE;
    VkPipeline overlayPipeline = VK_NULL_HANDLE, linePipeline = VK_NULL_HANDLE;
    VkDescriptorSet postDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet capturePostDescriptor = VK_NULL_HANDLE;
    VkImage shadowImage = VK_NULL_HANDLE;
    VkDeviceMemory shadowMemory = VK_NULL_HANDLE;
    VkImageView shadowView = VK_NULL_HANDLE;
    VkSampler shadowSampler = VK_NULL_HANDLE;
    VkRenderPass shadowRenderPass = VK_NULL_HANDLE;
    VkFramebuffer shadowFramebuffer = VK_NULL_HANDLE;
    VkShaderModule shadowVert = VK_NULL_HANDLE;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VkSampler nearestSampler = VK_NULL_HANDLE, linearSampler = VK_NULL_HANDLE;
    std::vector<VertexSlice> vertexSlices;
    uint32_t vertexSliceIndex = 0;
    VertexSlice* currentVertexSlice = nullptr;
    std::vector<float> rtFramePositions;
    std::vector<RayMaterial> rtFrameMaterials;
    RayScene rtCurrent{},rtPending{};
    bool rtPromotionPending = false;
    VkDescriptorSet rtDescriptor = VK_NULL_HANDLE;
    VkBuffer uniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory uniformMemory = VK_NULL_HANDLE;
    void* uniformMap = nullptr;
    VkBuffer vramStaging = VK_NULL_HANDLE;
    VkDeviceMemory vramStagingMemory = VK_NULL_HANDLE;
    void* vramStagingMap = nullptr;
    VkBuffer framebufferReadback = VK_NULL_HANDLE;
    VkDeviceMemory framebufferReadbackMemory = VK_NULL_HANDLE;
    void* framebufferReadbackMap = nullptr;
    VkBuffer overlayUpload = VK_NULL_HANDLE, overlayLines = VK_NULL_HANDLE;
    VkDeviceMemory overlayUploadMemory = VK_NULL_HANDLE, overlayLineMemory = VK_NULL_HANDLE;
    void* overlayUploadMap = nullptr;
    void* overlayLineMap = nullptr;
    VkDeviceSize overlayUploadCursor = 0;
    VkDeviceSize uniformStride = 0;
    uint32_t uniformSlot = 0;
    Texture textures[kMaxTextures];
    std::vector<Texture> retiredTextures;
    TextureID nextTexture = 1, boundTexture = 0;
    TexFormat texFormat = TF_16_BIT;
    std::vector<PipelineEntry> pipelines;

    Uniforms uniforms{};
    VkViewport viewport{};
    VkRect2D scissor{};
    bool scissorEnabled = false;
    BlendMode blend = BM_NONE;
    bool depth = false, depthAlways = false, stencil = false, wire = false;
    float polygonOffset = 0.0f;
    TextureID fmvTexture = 0;
    TextureID captureTexture = 0;
    TextureID framebufferTexture = 0;
    TextureID offscreenTexture = 0;
    Texture postTexture{};
    bool captureRequested = false;
    bool captureValid = false;
    bool postRequested = false;
    uint32_t postFrame = 0;
    bool framebufferStoreRequested = false, framebufferStoreValid = false;
    VkRect2D framebufferRect{};
    VkRect2D framebufferReadPendingRects[kFramebufferRegions]{};
    VkRect2D framebufferReadReadyRects[kFramebufferRegions]{};
    uint32_t framebufferReadPendingCount = 0, framebufferReadReadyCount = 0;
    VkRect2D displayFramebufferRect{};
    bool displayFramebufferRectValid = false;
    VkRect2D framebufferRects[kFramebufferRegions]{};
    uint32_t framebufferRectCount = 0, framebufferRectCursor = 0;
    RECT16 offscreenRect{};
    bool inOffscreen = false;
    bool inShadow = false;
    bool overlayPassSuspended = false;
    bool overlayLogged = false;
    float shadowLightMatrix[16] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    float shadowZNear = 20.0f, shadowZFar = 5200.0f;
    std::vector<uint8_t> fmvRgba;
} vk;

static bool Check(VkResult result, const char* what) {
    if (result == VK_SUCCESS) return true;
    eprinterr("Vulkan: %s failed (%d)\n", what, (int)result);
    return false;
}

static bool HasDeviceExtension(VkPhysicalDevice device, const char* name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) != VK_SUCCESS)
        return false;
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data()) != VK_SUCCESS)
        return false;
    for (const VkExtensionProperties& extension : extensions)
        if (strcmp(extension.extensionName, name) == 0) return true;
    return false;
}

static bool HasRayQueryExtensions(VkPhysicalDevice device) {
    static const char* required[] = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME
    };
    for (const char* extension : required)
        if (!HasDeviceExtension(device, extension)) return false;
    return true;
}

static void RememberFramebufferRect(int x,int y,int w,int h) {
    if(w<=0||h<=0)return;
    for(uint32_t i=0;i<vk.framebufferRectCount;i++){
        const VkRect2D& r=vk.framebufferRects[i];
        if(r.offset.x==x&&r.offset.y==y&&r.extent.width==(uint32_t)w&&r.extent.height==(uint32_t)h)return;
    }
    uint32_t slot;
    if(vk.framebufferRectCount<kFramebufferRegions)slot=vk.framebufferRectCount++;
    else{slot=vk.framebufferRectCursor;vk.framebufferRectCursor=(vk.framebufferRectCursor+1)%kFramebufferRegions;}
    vk.framebufferRects[slot]={{x,y},{(uint32_t)w,(uint32_t)h}};
    vk.framebufferStoreValid=true;
}

/* The normal display feedback buffer is a single moving VRAM rectangle.  The
 * OpenGL backend's g_PreviousFramebuffer replaces it on every store; retaining
 * both alternating display pages makes unrelated textures that later occupy an
 * old page sample stale screen imagery.  Offscreen render targets are tracked
 * separately above because more than one of those may remain resident. */
static void RememberDisplayFramebufferRect(int x,int y,int w,int h) {
    if(w<=0||h<=0){vk.displayFramebufferRectValid=false;return;}
    vk.displayFramebufferRect={{x,y},{(uint32_t)w,(uint32_t)h}};
    vk.displayFramebufferRectValid=true;
    vk.framebufferStoreValid=true;
}

static void QueueFramebufferReadRect(int x,int y,int w,int h) {
    if(w<=0||h<=0)return;
    for(uint32_t i=0;i<vk.framebufferReadPendingCount;i++){
        const VkRect2D& r=vk.framebufferReadPendingRects[i];
        if(r.offset.x==x&&r.offset.y==y&&r.extent.width==(uint32_t)w&&r.extent.height==(uint32_t)h)return;
    }
    if(vk.framebufferReadPendingCount<kFramebufferRegions)vk.framebufferReadPendingRects[vk.framebufferReadPendingCount++]={{x,y},{(uint32_t)w,(uint32_t)h}};
}

static void InvalidateFramebufferRects(int x,int y,int w,int h){
    if(w<=0||h<=0)return;
    for(uint32_t i=0;i<vk.framebufferRectCount;){const VkRect2D& r=vk.framebufferRects[i];bool overlap=x<(int)(r.offset.x+r.extent.width)&&x+w>r.offset.x&&y<(int)(r.offset.y+r.extent.height)&&y+h>r.offset.y;if(overlap)vk.framebufferRects[i]=vk.framebufferRects[--vk.framebufferRectCount];else i++;}vk.framebufferStoreValid=vk.framebufferRectCount>0;
    if(vk.displayFramebufferRectValid){const VkRect2D& r=vk.displayFramebufferRect;bool overlap=x<(int)(r.offset.x+r.extent.width)&&x+w>r.offset.x&&y<(int)(r.offset.y+r.extent.height)&&y+h>r.offset.y;if(overlap)vk.displayFramebufferRectValid=false;}
    vk.framebufferStoreValid=vk.displayFramebufferRectValid||vk.framebufferRectCount>0;
}

static uint32_t FindMemory(uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties p{};
    vkGetPhysicalDeviceMemoryProperties(vk.physical, &p);
    for (uint32_t i=0; i<p.memoryTypeCount; ++i)
        if ((bits & (1u<<i)) && (p.memoryTypes[i].propertyFlags & flags) == flags) return i;
    return UINT32_MAX;
}

static bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                         VkBuffer& buffer, VkDeviceMemory& memory, void** mapped = nullptr,
                         bool deviceAddress = false) {
    if (deviceAddress) usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size=size; bi.usage=usage;
    if (!Check(vkCreateBuffer(vk.device,&bi,nullptr,&buffer),"vkCreateBuffer")) return false;
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(vk.device,buffer,&mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=mr.size; ai.memoryTypeIndex=FindMemory(mr.memoryTypeBits,props);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if (deviceAddress) { flags.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT; ai.pNext=&flags; }
    if (ai.memoryTypeIndex==UINT32_MAX || !Check(vkAllocateMemory(vk.device,&ai,nullptr,&memory),"vkAllocateMemory")) {
        vkDestroyBuffer(vk.device,buffer,nullptr); buffer=VK_NULL_HANDLE; return false;
    }
    vkBindBufferMemory(vk.device,buffer,memory,0);
    if (mapped) Check(vkMapMemory(vk.device,memory,0,size,0,mapped),"vkMapMemory");
    return true;
}

static void DestroyRayScene(RayScene& scene) {
    if(scene.tlas)vk.destroyAccelerationStructure(vk.device,scene.tlas,nullptr);
    if(scene.blas)vk.destroyAccelerationStructure(vk.device,scene.blas,nullptr);
    VkBuffer buffers[]={scene.geometryBuffer,scene.materialBuffer,scene.blasBuffer,scene.blasScratch,scene.instanceBuffer,scene.tlasBuffer,scene.tlasScratch};
    VkDeviceMemory memories[]={scene.geometryMemory,scene.materialMemory,scene.blasMemory,scene.blasScratchMemory,scene.instanceMemory,scene.tlasMemory,scene.tlasScratchMemory};
    for(VkBuffer buffer:buffers)if(buffer)vkDestroyBuffer(vk.device,buffer,nullptr);
    for(VkDeviceMemory memory:memories)if(memory)vkFreeMemory(vk.device,memory,nullptr);
    scene=RayScene{};
}

static VkDeviceAddress BufferAddress(VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};info.buffer=buffer;
    return vk.getBufferDeviceAddress?vk.getBufferDeviceAddress(vk.device,&info):0;
}

static void PromoteRayScene() {
    if(!vk.rtPromotionPending)return;
    DestroyRayScene(vk.rtCurrent);vk.rtCurrent=vk.rtPending;vk.rtPending=RayScene{};vk.rtPromotionPending=false;
    if(vk.rtCurrent.tlas&&vk.rtDescriptor){
        VkWriteDescriptorSetAccelerationStructureKHR accelerationWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};accelerationWrite.accelerationStructureCount=1;accelerationWrite.pAccelerationStructures=&vk.rtCurrent.tlas;
        VkDescriptorBufferInfo materialInfo{vk.rtCurrent.materialBuffer,0,VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[2]{};writes[0].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[0].pNext=&accelerationWrite;writes[0].dstSet=vk.rtDescriptor;writes[0].dstBinding=0;writes[0].descriptorCount=1;writes[0].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[1].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[1].dstSet=vk.rtDescriptor;writes[1].dstBinding=1;writes[1].descriptorCount=1;writes[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[1].pBufferInfo=&materialInfo;vkUpdateDescriptorSets(vk.device,2,writes,0,nullptr);
    }
}

static bool BuildRaySceneForNextFrame(VkCommandBuffer command) {
    vk.rtPromotionPending=true;
    if(!g_cfg_rtgi||!vk.rayQueryAvailable||vk.rtFramePositions.size()<9)return false;
    DestroyRayScene(vk.rtPending);RayScene& scene=vk.rtPending;
    const uint64_t requestedTriangles=std::min(vk.rtFramePositions.size()/9,vk.rtFrameMaterials.size());
    const uint64_t maximumTriangles=std::max<uint64_t>(1,vk.accelerationProperties.maxPrimitiveCount);
    scene.triangleCount=(uint32_t)std::min(requestedTriangles,maximumTriangles);
    static bool firstSceneLogged=false;if(!firstSceneLogged){eprintf("*Vulkan RT: building live scene (%u triangles, one-frame history)\n",scene.triangleCount);firstSceneLogged=true;}
    const VkDeviceSize geometryBytes=(VkDeviceSize)scene.triangleCount*9*sizeof(float);
    void* geometryMap=nullptr;
    if(!CreateBuffer(geometryBytes,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,scene.geometryBuffer,scene.geometryMemory,&geometryMap,true))return false;
    memcpy(geometryMap,vk.rtFramePositions.data(),(size_t)geometryBytes);vkUnmapMemory(vk.device,scene.geometryMemory);
    const VkDeviceSize materialBytes=(VkDeviceSize)scene.triangleCount*sizeof(RayMaterial);void* materialMap=nullptr;
    if(!CreateBuffer(materialBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,scene.materialBuffer,scene.materialMemory,&materialMap)){DestroyRayScene(scene);return false;}
    memcpy(materialMap,vk.rtFrameMaterials.data(),(size_t)materialBytes);vkUnmapMemory(vk.device,scene.materialMemory);
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};triangles.vertexFormat=VK_FORMAT_R32G32B32_SFLOAT;triangles.vertexData.deviceAddress=BufferAddress(scene.geometryBuffer);triangles.vertexStride=3*sizeof(float);triangles.maxVertex=scene.triangleCount*3-1;
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};geometry.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR;geometry.flags=VK_GEOMETRY_OPAQUE_BIT_KHR;geometry.geometry.triangles=triangles;
    VkAccelerationStructureBuildGeometryInfoKHR blasBuild{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};blasBuild.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;blasBuild.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;blasBuild.mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;blasBuild.geometryCount=1;blasBuild.pGeometries=&geometry;
    VkAccelerationStructureBuildSizesInfoKHR blasSizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};uint32_t primitiveCount=scene.triangleCount;vk.getAccelerationStructureBuildSizes(vk.device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&blasBuild,&primitiveCount,&blasSizes);
    if(!CreateBuffer(blasSizes.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,scene.blasBuffer,scene.blasMemory,nullptr,true)||!CreateBuffer(blasSizes.buildScratchSize,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,scene.blasScratch,scene.blasScratchMemory,nullptr,true)){DestroyRayScene(scene);return false;}
    VkAccelerationStructureCreateInfoKHR blasCreate{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};blasCreate.buffer=scene.blasBuffer;blasCreate.size=blasSizes.accelerationStructureSize;blasCreate.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;if(!Check(vk.createAccelerationStructure(vk.device,&blasCreate,nullptr,&scene.blas),"RT BLAS")){DestroyRayScene(scene);return false;}
    blasBuild.dstAccelerationStructure=scene.blas;blasBuild.scratchData.deviceAddress=BufferAddress(scene.blasScratch);VkAccelerationStructureBuildRangeInfoKHR blasRange{scene.triangleCount,0,0,0};const VkAccelerationStructureBuildRangeInfoKHR* blasRanges=&blasRange;
    VkMemoryBarrier hostBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};hostBarrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;hostBarrier.dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,0,1,&hostBarrier,0,nullptr,0,nullptr);vk.cmdBuildAccelerationStructures(command,1,&blasBuild,&blasRanges);
    VkAccelerationStructureDeviceAddressInfoKHR blasAddressInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};blasAddressInfo.accelerationStructure=scene.blas;
    VkAccelerationStructureInstanceKHR instance{};instance.transform.matrix[0][0]=1.0f;instance.transform.matrix[1][1]=1.0f;instance.transform.matrix[2][2]=1.0f;instance.mask=0xff;instance.flags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;instance.accelerationStructureReference=vk.getAccelerationStructureDeviceAddress(vk.device,&blasAddressInfo);
    void* instanceMap=nullptr;if(!CreateBuffer(sizeof(instance),VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,scene.instanceBuffer,scene.instanceMemory,&instanceMap,true)){DestroyRayScene(scene);return false;}memcpy(instanceMap,&instance,sizeof(instance));vkUnmapMemory(vk.device,scene.instanceMemory);
    VkAccelerationStructureGeometryInstancesDataKHR instances{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};instances.arrayOfPointers=VK_FALSE;instances.data.deviceAddress=BufferAddress(scene.instanceBuffer);VkAccelerationStructureGeometryKHR tlasGeometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};tlasGeometry.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR;tlasGeometry.geometry.instances=instances;
    VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};tlasBuild.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;tlasBuild.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;tlasBuild.mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;tlasBuild.geometryCount=1;tlasBuild.pGeometries=&tlasGeometry;uint32_t instanceCount=1;VkAccelerationStructureBuildSizesInfoKHR tlasSizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};vk.getAccelerationStructureBuildSizes(vk.device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&tlasBuild,&instanceCount,&tlasSizes);
    if(!CreateBuffer(tlasSizes.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,scene.tlasBuffer,scene.tlasMemory,nullptr,true)||!CreateBuffer(tlasSizes.buildScratchSize,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,scene.tlasScratch,scene.tlasScratchMemory,nullptr,true)){DestroyRayScene(scene);return false;}
    VkAccelerationStructureCreateInfoKHR tlasCreate{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};tlasCreate.buffer=scene.tlasBuffer;tlasCreate.size=tlasSizes.accelerationStructureSize;tlasCreate.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;if(!Check(vk.createAccelerationStructure(vk.device,&tlasCreate,nullptr,&scene.tlas),"RT TLAS")){DestroyRayScene(scene);return false;}
    VkMemoryBarrier buildBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};buildBarrier.srcAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR|VK_ACCESS_HOST_WRITE_BIT;buildBarrier.dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR|VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,0,1,&buildBarrier,0,nullptr,0,nullptr);tlasBuild.dstAccelerationStructure=scene.tlas;tlasBuild.scratchData.deviceAddress=BufferAddress(scene.tlasScratch);VkAccelerationStructureBuildRangeInfoKHR tlasRange{1,0,0,0};const VkAccelerationStructureBuildRangeInfoKHR* tlasRanges=&tlasRange;vk.cmdBuildAccelerationStructures(command,1,&tlasBuild,&tlasRanges);
    VkMemoryBarrier traceBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};traceBarrier.srcAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;traceBarrier.dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,1,&traceBarrier,0,nullptr,0,nullptr);
    return true;
}

static void Immediate(const std::function<void(VkCommandBuffer)>& fn) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=vk.commandPool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
    VkCommandBuffer cmd{}; vkAllocateCommandBuffers(vk.device,&ai,&cmd);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd,&bi); fn(cmd); vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
    vkQueueSubmit(vk.queue,1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(vk.queue);
    vkFreeCommandBuffers(vk.device,vk.commandPool,1,&cmd);
}

static void Transition(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.oldLayout=oldLayout; b.newLayout=newLayout;
    b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=image;
    b.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT; b.subresourceRange.levelCount=1; b.subresourceRange.layerCount=1;
    if(oldLayout==VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
    else if(oldLayout==VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)b.srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    else if(oldLayout==VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)b.srcAccessMask=VK_ACCESS_SHADER_READ_BIT;
    else if(oldLayout==VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)b.srcAccessMask=VK_ACCESS_MEMORY_READ_BIT;
    if(newLayout==VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
    else if(newLayout==VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)b.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    else if(newLayout==VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    else if(newLayout==VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);
}

static bool UploadTexture(Texture& t, const void* pixels, size_t bytes, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags extraUsage=0) {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ii.imageType=VK_IMAGE_TYPE_2D; ii.format=format; ii.extent={w,h,1}; ii.mipLevels=1; ii.arrayLayers=1;
    ii.samples=VK_SAMPLE_COUNT_1_BIT; ii.tiling=VK_IMAGE_TILING_OPTIMAL; ii.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|extraUsage; ii.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
    if (!Check(vkCreateImage(vk.device,&ii,nullptr,&t.image),"vkCreateImage")) return false;
    VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(vk.device,t.image,&mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=mr.size; ai.memoryTypeIndex=FindMemory(mr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!Check(vkAllocateMemory(vk.device,&ai,nullptr,&t.memory),"texture memory")) return false;
    vkBindImageMemory(vk.device,t.image,t.memory,0);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image=t.image; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=format;
    vi.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT; vi.subresourceRange.levelCount=1; vi.subresourceRange.layerCount=1;
    Check(vkCreateImageView(vk.device,&vi,nullptr,&t.view),"texture view");

    VkBuffer staging{}; VkDeviceMemory stagingMem{}; void* map=nullptr;
    CreateBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,staging,stagingMem,&map);
    memcpy(map,pixels,bytes); vkUnmapMemory(vk.device,stagingMem);
    Immediate([&](VkCommandBuffer cmd){
        Transition(cmd,t.image,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy c{}; c.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT; c.imageSubresource.layerCount=1; c.imageExtent={w,h,1};
        vkCmdCopyBufferToImage(cmd,staging,t.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&c);
        Transition(cmd,t.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vkDestroyBuffer(vk.device,staging,nullptr); vkFreeMemory(vk.device,stagingMem,nullptr);
    t.width=w; t.height=h; t.alive=true;
    return true;
}

static void DestroyTexture(Texture& t) {
    if (!t.alive) return;
    VkDescriptorSet descriptor = t.descriptor;
    vkDestroyImageView(vk.device,t.view,nullptr); vkDestroyImage(vk.device,t.image,nullptr); vkFreeMemory(vk.device,t.memory,nullptr); t=Texture{};
    t.descriptor = descriptor;
}

static void ReleaseTextureForReplacement(Texture& t){
    if(!t.alive)return;
    /* Descriptor contents are consumed when the recorded command buffer runs,
     * not when vkCmdBindDescriptorSets is called.  Keep the old descriptor with
     * the retired image and allocate a fresh set for an in-frame replacement,
     * otherwise earlier draws silently start sampling the new texture. */
    if(vk.recording){vk.retiredTextures.push_back(t);t=Texture{};}
    else{vkWaitForFences(vk.device,1,&vk.fence,VK_TRUE,UINT64_MAX);DestroyTexture(t);}
}

static void DestroyRetiredTextures(){for(Texture& texture:vk.retiredTextures)DestroyTexture(texture);vk.retiredTextures.clear();}

static void UpdateDescriptor(TextureID id) {
    Texture& t=vk.textures[id];
    if (!t.descriptor) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool=vk.descriptorPool; ai.descriptorSetCount=1; ai.pSetLayouts=&vk.setLayout;
        Check(vkAllocateDescriptorSets(vk.device,&ai,&t.descriptor),"descriptor set");
    }
    VkDescriptorBufferInfo ub{vk.uniformBuffer,0,sizeof(Uniforms)};
    VkDescriptorImageInfo im{t.linear?vk.linearSampler:vk.nearestSampler,t.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    Texture& fb=(vk.framebufferTexture&&vk.textures[vk.framebufferTexture].alive)?vk.textures[vk.framebufferTexture]:t;
    VkDescriptorImageInfo fbIm{vk.nearestSampler,fb.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo shadowIm{vk.shadowSampler?vk.shadowSampler:vk.nearestSampler,vk.shadowView?vk.shadowView:t.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w[4]{};
    w[0]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[0].dstSet=t.descriptor; w[0].dstBinding=0; w[0].descriptorCount=1; w[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; w[0].pBufferInfo=&ub;
    w[1]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[1].dstSet=t.descriptor; w[1].dstBinding=1; w[1].descriptorCount=1; w[1].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo=&im;
    w[2]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[2].dstSet=t.descriptor; w[2].dstBinding=2; w[2].descriptorCount=1; w[2].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[2].pImageInfo=&fbIm;
    w[3]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[3].dstSet=t.descriptor; w[3].dstBinding=3; w[3].descriptorCount=1; w[3].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[3].pImageInfo=&shadowIm;
    vkUpdateDescriptorSets(vk.device,4,w,0,nullptr);
}

static VkShaderModule Shader(const uint32_t* data, size_t size) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; ci.codeSize=size; ci.pCode=data;
    VkShaderModule m{}; Check(vkCreateShaderModule(vk.device,&ci,nullptr,&m),"shader module"); return m;
}

static VkPipeline GetPipeline() {
    const bool useRayTracing=GR_RayTracingEnabled()&&vk.rtCurrent.tlas&&!vk.inOffscreen&&vk.rtFrag&&vk.rtPipelineLayout;
    PipelineKey key{(uint8_t)vk.blend,(uint8_t)vk.depth,(uint8_t)vk.depthAlways,(uint8_t)vk.stencil,(uint8_t)vk.wire,(uint8_t)vk.inOffscreen,(uint8_t)useRayTracing};
    for (auto& p:vk.pipelines) if (p.key==key) return p.pipeline;
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vk.vert; stages[0].pName="main";
    stages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=useRayTracing?vk.rtFrag:vk.frag; stages[1].pName="main";
    VkVertexInputBindingDescription bind{0,sizeof(GrVertex),VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription a[]={{0,0,VK_FORMAT_R16G16_SINT,0},{1,0,VK_FORMAT_R16G16_SINT,4},{2,0,VK_FORMAT_R32_SFLOAT,8},
      {3,0,VK_FORMAT_R8G8B8A8_UINT,12},{4,0,VK_FORMAT_R8G8B8A8_UNORM,16},{5,0,VK_FORMAT_R8G8B8A8_SINT,20},
      {6,0,VK_FORMAT_R32G32B32_SFLOAT,24},{7,0,VK_FORMAT_R32G32B32_SFLOAT,36},{8,0,VK_FORMAT_R32G32B32_SFLOAT,48}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO}; vi.vertexBindingDescriptionCount=1; vi.pVertexBindingDescriptions=&bind; vi.vertexAttributeDescriptionCount=9; vi.pVertexAttributeDescriptions=a;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; vp.viewportCount=1; vp.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.polygonMode=vk.wire?VK_POLYGON_MODE_LINE:VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_NONE; rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.depthBiasEnable=VK_TRUE; rs.lineWidth=1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples=vk.inOffscreen?VK_SAMPLE_COUNT_1_BIT:vk.samples;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}; ds.depthTestEnable=vk.depth; ds.depthWriteEnable=vk.depth; ds.depthCompareOp=vk.depthAlways?VK_COMPARE_OP_ALWAYS:VK_COMPARE_OP_LESS_OR_EQUAL; ds.stencilTestEnable=VK_TRUE;
    VkStencilOpState st{}; st.compareOp=vk.stencil?VK_COMPARE_OP_ALWAYS:VK_COMPARE_OP_NOT_EQUAL; st.passOp=vk.stencil?VK_STENCIL_OP_REPLACE:VK_STENCIL_OP_KEEP; st.failOp=VK_STENCIL_OP_REPLACE; st.depthFailOp=vk.stencil?VK_STENCIL_OP_REPLACE:VK_STENCIL_OP_KEEP; st.compareMask=vk.stencil?0x10:0xff; st.writeMask=0xff; st.reference=1; ds.front=ds.back=st;
    VkPipelineColorBlendAttachmentState ba{}; ba.colorWriteMask=0xf; ba.blendEnable=vk.blend!=BM_NONE;
    if (vk.blend==BM_AVERAGE) { ba.srcColorBlendFactor=VK_BLEND_FACTOR_SRC_ALPHA; ba.dstColorBlendFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; }
    else if(vk.blend==BM_ADD_QUATER_SOURCE) { ba.srcColorBlendFactor=VK_BLEND_FACTOR_CONSTANT_ALPHA; ba.dstColorBlendFactor=VK_BLEND_FACTOR_ONE; }
    else { ba.srcColorBlendFactor=VK_BLEND_FACTOR_ONE; ba.dstColorBlendFactor=VK_BLEND_FACTOR_ONE; }
    ba.colorBlendOp=vk.blend==BM_SUBTRACT?VK_BLEND_OP_REVERSE_SUBTRACT:VK_BLEND_OP_ADD; ba.srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE; ba.dstAlphaBlendFactor=VK_BLEND_FACTOR_ZERO; ba.alphaBlendOp=VK_BLEND_OP_ADD;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cb.attachmentCount=1; cb.pAttachments=&ba; cb.blendConstants[3]=0.25f;
    VkDynamicState dyns[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR,VK_DYNAMIC_STATE_BLEND_CONSTANTS,VK_DYNAMIC_STATE_DEPTH_BIAS};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dy.dynamicStateCount=4; dy.pDynamicStates=dyns;
    VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; pi.stageCount=2; pi.pStages=stages; pi.pVertexInputState=&vi; pi.pInputAssemblyState=&ia; pi.pViewportState=&vp; pi.pRasterizationState=&rs; pi.pMultisampleState=&ms; pi.pDepthStencilState=&ds; pi.pColorBlendState=&cb; pi.pDynamicState=&dy; pi.layout=useRayTracing?vk.rtPipelineLayout:vk.pipelineLayout; pi.renderPass=vk.inOffscreen?vk.offscreenRenderPass:vk.renderPass;
    PipelineEntry e{}; e.key=key; Check(vkCreateGraphicsPipelines(vk.device,VK_NULL_HANDLE,1,&pi,nullptr,&e.pipeline),"graphics pipeline"); vk.pipelines.push_back(e); return e.pipeline;
}

static bool CreatePostPipeline() {
    if(vk.postPipeline)return true;
    if(!vk.postVert||!vk.postFrag||!vk.postPipelineLayout||!vk.loadRenderPass)return false;
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT;stages[0].module=vk.postVert;stages[0].pName="main";
    stages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT;stages[1].module=vk.postFrag;stages[1].pName="main";
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};vp.viewportCount=1;vp.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};rs.polygonMode=VK_POLYGON_MODE_FILL;rs.cullMode=VK_CULL_MODE_NONE;rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;rs.lineWidth=1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};ms.rasterizationSamples=vk.samples;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState ba{};ba.colorWriteMask=0xf;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};cb.attachmentCount=1;cb.pAttachments=&ba;
    VkDynamicState dynamics[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dy.dynamicStateCount=2;dy.pDynamicStates=dynamics;
    VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};pi.stageCount=2;pi.pStages=stages;pi.pVertexInputState=&vi;pi.pInputAssemblyState=&ia;pi.pViewportState=&vp;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pDepthStencilState=&ds;pi.pColorBlendState=&cb;pi.pDynamicState=&dy;pi.layout=vk.postPipelineLayout;pi.renderPass=vk.loadRenderPass;
    return Check(vkCreateGraphicsPipelines(vk.device,VK_NULL_HANDLE,1,&pi,nullptr,&vk.postPipeline),"post-process pipeline");
}

static bool CreateOverlayPipelines() {
    if(vk.overlayPipeline&&vk.linePipeline)return true;
    if(!vk.overlayVert||!vk.overlayFrag||!vk.lineVert||!vk.lineFrag||!vk.pipelineLayout||!vk.renderPass)return false;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};viewport.viewportCount=1;viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};raster.polygonMode=VK_POLYGON_MODE_FILL;raster.cullMode=VK_CULL_MODE_NONE;raster.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;raster.lineWidth=1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};multisample.rasterizationSamples=vk.samples;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkDynamicState states[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dynamic.dynamicStateCount=2;dynamic.pDynamicStates=states;
    VkPipelineColorBlendAttachmentState blend{};blend.blendEnable=VK_TRUE;blend.srcColorBlendFactor=VK_BLEND_FACTOR_SRC_ALPHA;blend.dstColorBlendFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;blend.colorBlendOp=VK_BLEND_OP_ADD;blend.srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE;blend.dstAlphaBlendFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;blend.alphaBlendOp=VK_BLEND_OP_ADD;blend.colorWriteMask=0xf;
    VkPipelineColorBlendStateCreateInfo color{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};color.attachmentCount=1;color.pAttachments=&blend;
    VkPipelineVertexInputStateCreateInfo noVertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};VkPipelineInputAssemblyStateCreateInfo triangles{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};triangles.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineShaderStageCreateInfo overlayStages[2]{};overlayStages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};overlayStages[0].stage=VK_SHADER_STAGE_VERTEX_BIT;overlayStages[0].module=vk.overlayVert;overlayStages[0].pName="main";overlayStages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};overlayStages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT;overlayStages[1].module=vk.overlayFrag;overlayStages[1].pName="main";
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};info.stageCount=2;info.pStages=overlayStages;info.pVertexInputState=&noVertices;info.pInputAssemblyState=&triangles;info.pViewportState=&viewport;info.pRasterizationState=&raster;info.pMultisampleState=&multisample;info.pDepthStencilState=&depth;info.pColorBlendState=&color;info.pDynamicState=&dynamic;info.layout=vk.pipelineLayout;info.renderPass=vk.renderPass;
    if(!Check(vkCreateGraphicsPipelines(vk.device,VK_NULL_HANDLE,1,&info,nullptr,&vk.overlayPipeline),"overlay pipeline"))return false;
    VkVertexInputBindingDescription binding{0,5*sizeof(float),VK_VERTEX_INPUT_RATE_VERTEX};VkVertexInputAttributeDescription attrs[2]={{0,0,VK_FORMAT_R32G32_SFLOAT,0},{1,0,VK_FORMAT_R32G32B32_SFLOAT,2*sizeof(float)}};VkPipelineVertexInputStateCreateInfo lineInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};lineInput.vertexBindingDescriptionCount=1;lineInput.pVertexBindingDescriptions=&binding;lineInput.vertexAttributeDescriptionCount=2;lineInput.pVertexAttributeDescriptions=attrs;VkPipelineInputAssemblyStateCreateInfo lines{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};lines.topology=VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    VkPipelineShaderStageCreateInfo lineStages[2]{};lineStages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};lineStages[0].stage=VK_SHADER_STAGE_VERTEX_BIT;lineStages[0].module=vk.lineVert;lineStages[0].pName="main";lineStages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};lineStages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT;lineStages[1].module=vk.lineFrag;lineStages[1].pName="main";blend.blendEnable=VK_FALSE;info.pStages=lineStages;info.pVertexInputState=&lineInput;info.pInputAssemblyState=&lines;
    if(!Check(vkCreateGraphicsPipelines(vk.device,VK_NULL_HANDLE,1,&info,nullptr,&vk.linePipeline),"overlay line pipeline"))return false;
    return true;
}

static bool CreateCompatibleRenderPass(VkAttachmentLoadOp colorLoad,
    VkImageLayout colorInitial,VkImageLayout colorFinal,
    VkAttachmentLoadOp depthLoad,VkRenderPass* output){
    VkAttachmentDescription at[2]{};at[0].format=vk.colorFormat;at[0].samples=VK_SAMPLE_COUNT_1_BIT;at[0].loadOp=colorLoad;at[0].storeOp=VK_ATTACHMENT_STORE_OP_STORE;at[0].initialLayout=colorInitial;at[0].finalLayout=colorFinal;
    at[1].format=vk.depthFormat;at[1].samples=VK_SAMPLE_COUNT_1_BIT;at[1].loadOp=depthLoad;at[1].storeOp=VK_ATTACHMENT_STORE_OP_STORE;at[1].stencilLoadOp=depthLoad;at[1].stencilStoreOp=VK_ATTACHMENT_STORE_OP_STORE;at[1].initialLayout=depthLoad==VK_ATTACHMENT_LOAD_OP_LOAD?VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_UNDEFINED;at[1].finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference cr{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},dr{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};VkSubpassDescription sp{};sp.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sp.colorAttachmentCount=1;sp.pColorAttachments=&cr;sp.pDepthStencilAttachment=&dr;
    VkSubpassDependency dep{};dep.srcSubpass=VK_SUBPASS_EXTERNAL;dep.dstSubpass=0;dep.srcStageMask=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;dep.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;dep.srcAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;dep.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};rp.attachmentCount=2;rp.pAttachments=at;rp.subpassCount=1;rp.pSubpasses=&sp;rp.dependencyCount=1;rp.pDependencies=&dep;return Check(vkCreateRenderPass(vk.device,&rp,nullptr,output),"render pass");
}

static bool CreateMainRenderPass(bool load,VkRenderPass* output){
    if(vk.samples==VK_SAMPLE_COUNT_1_BIT)return CreateCompatibleRenderPass(load?VK_ATTACHMENT_LOAD_OP_LOAD:VK_ATTACHMENT_LOAD_OP_CLEAR,load?VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,load?VK_ATTACHMENT_LOAD_OP_LOAD:VK_ATTACHMENT_LOAD_OP_CLEAR,output);
    VkAttachmentDescription attachments[3]{};
    attachments[0].format=vk.colorFormat;attachments[0].samples=vk.samples;attachments[0].loadOp=load?VK_ATTACHMENT_LOAD_OP_LOAD:VK_ATTACHMENT_LOAD_OP_CLEAR;attachments[0].storeOp=VK_ATTACHMENT_STORE_OP_STORE;attachments[0].initialLayout=load?VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_UNDEFINED;attachments[0].finalLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[1].format=vk.depthFormat;attachments[1].samples=vk.samples;attachments[1].loadOp=load?VK_ATTACHMENT_LOAD_OP_LOAD:VK_ATTACHMENT_LOAD_OP_CLEAR;attachments[1].storeOp=VK_ATTACHMENT_STORE_OP_STORE;attachments[1].stencilLoadOp=load?VK_ATTACHMENT_LOAD_OP_LOAD:VK_ATTACHMENT_LOAD_OP_CLEAR;attachments[1].stencilStoreOp=VK_ATTACHMENT_STORE_OP_STORE;attachments[1].initialLayout=load?VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_UNDEFINED;attachments[1].finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[2].format=vk.colorFormat;attachments[2].samples=VK_SAMPLE_COUNT_1_BIT;attachments[2].loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;attachments[2].storeOp=VK_ATTACHMENT_STORE_OP_STORE;attachments[2].initialLayout=load?VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:VK_IMAGE_LAYOUT_UNDEFINED;attachments[2].finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference color{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},depth{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},resolve{2,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};VkSubpassDescription subpass{};subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;subpass.colorAttachmentCount=1;subpass.pColorAttachments=&color;subpass.pResolveAttachments=&resolve;subpass.pDepthStencilAttachment=&depth;VkSubpassDependency dependency{};dependency.srcSubpass=VK_SUBPASS_EXTERNAL;dependency.dstSubpass=0;dependency.srcStageMask=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;dependency.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;dependency.srcAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;dependency.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};info.attachmentCount=3;info.pAttachments=attachments;info.subpassCount=1;info.pSubpasses=&subpass;info.dependencyCount=1;info.pDependencies=&dependency;return Check(vkCreateRenderPass(vk.device,&info,nullptr,output),"MSAA render pass");
}

static void DestroySwapchain() {
    if (!vk.device) return; vkDeviceWaitIdle(vk.device);
    if(vk.postPipeline)vkDestroyPipeline(vk.device,vk.postPipeline,nullptr);vk.postPipeline=VK_NULL_HANDLE;
    if(vk.overlayPipeline)vkDestroyPipeline(vk.device,vk.overlayPipeline,nullptr);vk.overlayPipeline=VK_NULL_HANDLE;
    if(vk.linePipeline)vkDestroyPipeline(vk.device,vk.linePipeline,nullptr);vk.linePipeline=VK_NULL_HANDLE;
    for(auto p:vk.pipelines) vkDestroyPipeline(vk.device,p.pipeline,nullptr); vk.pipelines.clear();
    for(auto f:vk.framebuffers) vkDestroyFramebuffer(vk.device,f,nullptr); vk.framebuffers.clear();
    if(vk.depthView) vkDestroyImageView(vk.device,vk.depthView,nullptr); if(vk.depthImage) vkDestroyImage(vk.device,vk.depthImage,nullptr); if(vk.depthMemory) vkFreeMemory(vk.device,vk.depthMemory,nullptr);vk.depthView=VK_NULL_HANDLE;vk.depthImage=VK_NULL_HANDLE;vk.depthMemory=VK_NULL_HANDLE;
    if(vk.msaaColorView)vkDestroyImageView(vk.device,vk.msaaColorView,nullptr);if(vk.msaaColorImage)vkDestroyImage(vk.device,vk.msaaColorImage,nullptr);if(vk.msaaColorMemory)vkFreeMemory(vk.device,vk.msaaColorMemory,nullptr);vk.msaaColorView=VK_NULL_HANDLE;vk.msaaColorImage=VK_NULL_HANDLE;vk.msaaColorMemory=VK_NULL_HANDLE;
    if(vk.renderPass) vkDestroyRenderPass(vk.device,vk.renderPass,nullptr);if(vk.loadRenderPass)vkDestroyRenderPass(vk.device,vk.loadRenderPass,nullptr);vk.renderPass=vk.loadRenderPass=VK_NULL_HANDLE;for(auto v:vk.views) vkDestroyImageView(vk.device,v,nullptr); vk.views.clear();
    if(vk.swapchain) vkDestroySwapchainKHR(vk.device,vk.swapchain,nullptr); vk.swapchain=VK_NULL_HANDLE;
}

static bool CreateSwapchain() {
    VkSurfaceCapabilitiesKHR caps{}; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical,vk.surface,&caps);
    uint32_t n=0; vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical,vk.surface,&n,nullptr); std::vector<VkSurfaceFormatKHR> formats(n); vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical,vk.surface,&n,formats.data());
    if(formats.empty()){eprinterr("Vulkan: surface exposes no color formats\n");return false;}
    VkSurfaceFormatKHR sf=formats[0]; for(auto f:formats) if(f.format==VK_FORMAT_B8G8R8A8_UNORM){sf=f;break;} vk.colorFormat=sf.format;
    if(caps.currentExtent.width!=UINT32_MAX) vk.extent=caps.currentExtent; else { int w,h; SDL_Vulkan_GetDrawableSize(g_window,&w,&h); vk.extent={(uint32_t)w,(uint32_t)h}; }
    if(vk.extent.width==0||vk.extent.height==0){vk.recreate=true;return false;}
    vk.extent.width=std::clamp(vk.extent.width,caps.minImageExtent.width,caps.maxImageExtent.width);vk.extent.height=std::clamp(vk.extent.height,caps.minImageExtent.height,caps.maxImageExtent.height);
    if((caps.supportedUsageFlags&(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT))!=(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT)){eprinterr("Vulkan: surface cannot provide a renderable/readable swapchain\n");return false;}
    uint32_t count=std::max(2u,caps.minImageCount); if(caps.maxImageCount) count=std::min(count,caps.maxImageCount);
    uint32_t pmCount=0;vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physical,vk.surface,&pmCount,nullptr);std::vector<VkPresentModeKHR> modes(pmCount);vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physical,vk.surface,&pmCount,modes.data());VkPresentModeKHR chosen=VK_PRESENT_MODE_FIFO_KHR;if(vk.swapInterval<=0){for(VkPresentModeKHR mode:modes)if(mode==VK_PRESENT_MODE_MAILBOX_KHR){chosen=mode;break;}if(chosen==VK_PRESENT_MODE_FIFO_KHR)for(VkPresentModeKHR mode:modes)if(mode==VK_PRESENT_MODE_IMMEDIATE_KHR){chosen=mode;break;}}
    VkCompositeAlphaFlagBitsKHR composite=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;for(VkCompositeAlphaFlagBitsKHR candidate:{VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})if(caps.supportedCompositeAlpha&candidate){composite=candidate;break;}
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR}; ci.surface=vk.surface; ci.minImageCount=count; ci.imageFormat=sf.format; ci.imageColorSpace=sf.colorSpace; ci.imageExtent=vk.extent; ci.imageArrayLayers=1; ci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT; ci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE; ci.preTransform=caps.currentTransform; ci.compositeAlpha=composite; ci.presentMode=chosen; ci.clipped=VK_TRUE;
    if(!Check(vkCreateSwapchainKHR(vk.device,&ci,nullptr,&vk.swapchain),"swapchain")) return false;
    vkGetSwapchainImagesKHR(vk.device,vk.swapchain,&count,nullptr); vk.images.resize(count); vkGetSwapchainImagesKHR(vk.device,vk.swapchain,&count,vk.images.data()); vk.views.resize(count);
    for(uint32_t i=0;i<count;i++){ VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image=vk.images[i]; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=vk.colorFormat; vi.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT; vi.subresourceRange.levelCount=1;vi.subresourceRange.layerCount=1; Check(vkCreateImageView(vk.device,&vi,nullptr,&vk.views[i]),"swap view"); }
    CreateMainRenderPass(false,&vk.renderPass);CreateMainRenderPass(true,&vk.loadRenderPass);
    VkImageCreateInfo di{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};di.imageType=VK_IMAGE_TYPE_2D;di.format=vk.depthFormat;di.extent={vk.extent.width,vk.extent.height,1};di.mipLevels=1;di.arrayLayers=1;di.samples=vk.samples;di.tiling=VK_IMAGE_TILING_OPTIMAL;di.usage=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;vkCreateImage(vk.device,&di,nullptr,&vk.depthImage); VkMemoryRequirements mr{};vkGetImageMemoryRequirements(vk.device,vk.depthImage,&mr);VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=mr.size;ai.memoryTypeIndex=FindMemory(mr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);vkAllocateMemory(vk.device,&ai,nullptr,&vk.depthMemory);vkBindImageMemory(vk.device,vk.depthImage,vk.depthMemory,0);VkImageViewCreateInfo dv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};dv.image=vk.depthImage;dv.viewType=VK_IMAGE_VIEW_TYPE_2D;dv.format=vk.depthFormat;dv.subresourceRange.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT;dv.subresourceRange.levelCount=1;dv.subresourceRange.layerCount=1;vkCreateImageView(vk.device,&dv,nullptr,&vk.depthView);
    if(vk.samples!=VK_SAMPLE_COUNT_1_BIT){VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};ci.imageType=VK_IMAGE_TYPE_2D;ci.format=vk.colorFormat;ci.extent={vk.extent.width,vk.extent.height,1};ci.mipLevels=1;ci.arrayLayers=1;ci.samples=vk.samples;ci.tiling=VK_IMAGE_TILING_OPTIMAL;ci.usage=VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;vkCreateImage(vk.device,&ci,nullptr,&vk.msaaColorImage);VkMemoryRequirements cmr{};vkGetImageMemoryRequirements(vk.device,vk.msaaColorImage,&cmr);VkMemoryAllocateInfo cai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};cai.allocationSize=cmr.size;cai.memoryTypeIndex=FindMemory(cmr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);vkAllocateMemory(vk.device,&cai,nullptr,&vk.msaaColorMemory);vkBindImageMemory(vk.device,vk.msaaColorImage,vk.msaaColorMemory,0);VkImageViewCreateInfo cv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};cv.image=vk.msaaColorImage;cv.viewType=VK_IMAGE_VIEW_TYPE_2D;cv.format=vk.colorFormat;cv.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;cv.subresourceRange.levelCount=1;cv.subresourceRange.layerCount=1;vkCreateImageView(vk.device,&cv,nullptr,&vk.msaaColorView);}
    vk.framebuffers.resize(count);for(uint32_t i=0;i<count;i++){VkImageView av[3]={vk.samples==VK_SAMPLE_COUNT_1_BIT?vk.views[i]:vk.msaaColorView,vk.depthView,vk.views[i]};VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};fi.renderPass=vk.renderPass;fi.attachmentCount=vk.samples==VK_SAMPLE_COUNT_1_BIT?2u:3u;fi.pAttachments=av;fi.width=vk.extent.width;fi.height=vk.extent.height;fi.layers=1;vkCreateFramebuffer(vk.device,&fi,nullptr,&vk.framebuffers[i]);}
    g_windowWidth=(int)vk.extent.width;g_windowHeight=(int)vk.extent.height;vk.viewport={0,0,(float)vk.extent.width,(float)vk.extent.height,0,1};vk.scissor={{0,0},vk.extent};if(vk.postPipelineLayout&&!CreatePostPipeline())return false;if(vk.overlayVert&&!CreateOverlayPipelines())return false;return true;
}

static bool InitDevice() {
    unsigned extCount=0;SDL_Vulkan_GetInstanceExtensions(g_window,&extCount,nullptr);std::vector<const char*> exts(extCount);SDL_Vulkan_GetInstanceExtensions(g_window,&extCount,exts.data());
    uint32_t availableCount=0;vkEnumerateInstanceExtensionProperties(nullptr,&availableCount,nullptr);std::vector<VkExtensionProperties> availableExtensions(availableCount);vkEnumerateInstanceExtensionProperties(nullptr,&availableCount,availableExtensions.data());
    for(const VkExtensionProperties& extension:availableExtensions)if(strcmp(extension.extensionName,VK_EXT_DEBUG_UTILS_EXTENSION_NAME)==0){exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);break;}
    extCount=(unsigned)exts.size();
    uint32_t loaderVersion=VK_API_VERSION_1_0;
    if(vkEnumerateInstanceVersion(&loaderVersion)!=VK_SUCCESS)loaderVersion=VK_API_VERSION_1_0;
    uint32_t requestedApi=loaderVersion>=VK_API_VERSION_1_2?VK_API_VERSION_1_2:
                          loaderVersion>=VK_API_VERSION_1_1?VK_API_VERSION_1_1:VK_API_VERSION_1_0;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="Silent Hill PC";app.apiVersion=requestedApi;VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ii.pApplicationInfo=&app;ii.enabledExtensionCount=extCount;ii.ppEnabledExtensionNames=exts.data();if(!Check(vkCreateInstance(&ii,nullptr,&vk.instance),"instance"))return false;
    if(!SDL_Vulkan_CreateSurface(g_window,vk.instance,&vk.surface)){eprinterr("SDL Vulkan surface: %s\n",SDL_GetError());return false;}
    uint32_t count=0;vkEnumeratePhysicalDevices(vk.instance,&count,nullptr);std::vector<VkPhysicalDevice> devs(count);vkEnumeratePhysicalDevices(vk.instance,&count,devs.data());
    int bestScore=-1;
    for(auto d:devs){
        uint32_t qn=0;vkGetPhysicalDeviceQueueFamilyProperties(d,&qn,nullptr);std::vector<VkQueueFamilyProperties> qq(qn);vkGetPhysicalDeviceQueueFamilyProperties(d,&qn,qq.data());
        for(uint32_t q=0;q<qn;q++){
            VkBool32 present=0;vkGetPhysicalDeviceSurfaceSupportKHR(d,q,vk.surface,&present);
            if(!present||!(qq[q].queueFlags&VK_QUEUE_GRAPHICS_BIT))continue;
            VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(d,&properties);
            int score=HasRayQueryExtensions(d)?10000:0;
            if(properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)score+=1000;
            else if(properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)score+=100;
            if(score>bestScore){bestScore=score;vk.physical=d;vk.queueFamily=q;}
            break;
        }
    }
    if(!vk.physical){eprinterr("No Vulkan graphics/present device\n");return false;}
    vk.depthFormat=VK_FORMAT_UNDEFINED;for(VkFormat candidate:{VK_FORMAT_D24_UNORM_S8_UINT,VK_FORMAT_D32_SFLOAT_S8_UINT,VK_FORMAT_D16_UNORM_S8_UINT}){VkFormatProperties properties{};vkGetPhysicalDeviceFormatProperties(vk.physical,candidate,&properties);if(properties.optimalTilingFeatures&VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT){vk.depthFormat=candidate;break;}}if(vk.depthFormat==VK_FORMAT_UNDEFINED){eprinterr("Vulkan: no depth/stencil attachment format available\n");return false;}
    VkPhysicalDeviceProperties deviceProperties{};vkGetPhysicalDeviceProperties(vk.physical,&deviceProperties);VkSampleCountFlags supported=deviceProperties.limits.framebufferColorSampleCounts&deviceProperties.limits.framebufferDepthSampleCounts;int requested=g_cfg_msaaSamples;vk.samples=VK_SAMPLE_COUNT_1_BIT;if(requested>=8&&(supported&VK_SAMPLE_COUNT_8_BIT))vk.samples=VK_SAMPLE_COUNT_8_BIT;else if(requested>=4&&(supported&VK_SAMPLE_COUNT_4_BIT))vk.samples=VK_SAMPLE_COUNT_4_BIT;else if(requested>=2&&(supported&VK_SAMPLE_COUNT_2_BIT))vk.samples=VK_SAMPLE_COUNT_2_BIT;g_cfg_msaaSamples=vk.samples==VK_SAMPLE_COUNT_8_BIT?8:vk.samples==VK_SAMPLE_COUNT_4_BIT?4:vk.samples==VK_SAMPLE_COUNT_2_BIT?2:0;eprintf("*Vulkan MSAA: %dx\n",g_cfg_msaaSamples);
    float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=vk.queueFamily;qi.queueCount=1;qi.pQueuePriorities=&priority;
    VkPhysicalDeviceFeatures supportedFeatures{};vkGetPhysicalDeviceFeatures(vk.physical,&supportedFeatures);VkPhysicalDeviceFeatures features{};features.fillModeNonSolid=supportedFeatures.fillModeNonSolid;vk.fillModeNonSolid=features.fillModeNonSolid==VK_TRUE;
    std::vector<const char*> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    const bool rayExtensions=HasRayQueryExtensions(vk.physical);
    bda.pNext=&acceleration;acceleration.pNext=&rayQuery;
    VkPhysicalDeviceFeatures2 featureQuery{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};featureQuery.pNext=rayExtensions?&bda:nullptr;vkGetPhysicalDeviceFeatures2(vk.physical,&featureQuery);
    vk.rayQueryAvailable=rayExtensions&&bda.bufferDeviceAddress&&acceleration.accelerationStructure&&rayQuery.rayQuery;
    void* deviceFeatureChain=nullptr;
    if(vk.rayQueryAvailable){
        static const char* rayExtensionsToEnable[]={
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,VK_KHR_RAY_QUERY_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_SPIRV_1_4_EXTENSION_NAME,VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME};
        deviceExtensions.insert(deviceExtensions.end(),std::begin(rayExtensionsToEnable),std::end(rayExtensionsToEnable));
        bda.bufferDeviceAddress=VK_TRUE;acceleration.accelerationStructure=VK_TRUE;rayQuery.rayQuery=VK_TRUE;deviceFeatureChain=&bda;
    }else{
        snprintf(vk.rayStatus,sizeof(vk.rayStatus),"unavailable (ray-query extensions/features missing)");
    }
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.pNext=deviceFeatureChain;di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;di.enabledExtensionCount=(uint32_t)deviceExtensions.size();di.ppEnabledExtensionNames=deviceExtensions.data();di.pEnabledFeatures=&features;if(!Check(vkCreateDevice(vk.physical,&di,nullptr,&vk.device),"device"))return false;
    vkGetDeviceQueue(vk.device,vk.queueFamily,0,&vk.queue);vk.cmdBeginDebugLabel=(PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(vk.device,"vkCmdBeginDebugUtilsLabelEXT");vk.cmdEndDebugLabel=(PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(vk.device,"vkCmdEndDebugUtilsLabelEXT");
    if(vk.rayQueryAvailable){
        vk.getBufferDeviceAddress=(PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(vk.device,"vkGetBufferDeviceAddressKHR");
        if(!vk.getBufferDeviceAddress)vk.getBufferDeviceAddress=(PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(vk.device,"vkGetBufferDeviceAddress");
        vk.createAccelerationStructure=(PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(vk.device,"vkCreateAccelerationStructureKHR");
        vk.destroyAccelerationStructure=(PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(vk.device,"vkDestroyAccelerationStructureKHR");
        vk.getAccelerationStructureBuildSizes=(PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(vk.device,"vkGetAccelerationStructureBuildSizesKHR");
        vk.cmdBuildAccelerationStructures=(PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(vk.device,"vkCmdBuildAccelerationStructuresKHR");
        vk.getAccelerationStructureDeviceAddress=(PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(vk.device,"vkGetAccelerationStructureDeviceAddressKHR");
        if(!vk.getBufferDeviceAddress||!vk.createAccelerationStructure||!vk.destroyAccelerationStructure||!vk.getAccelerationStructureBuildSizes||!vk.cmdBuildAccelerationStructures||!vk.getAccelerationStructureDeviceAddress){
            vk.rayQueryAvailable=false;snprintf(vk.rayStatus,sizeof(vk.rayStatus),"unavailable (driver entry points missing)");
        }else{
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};properties.pNext=&vk.accelerationProperties;vkGetPhysicalDeviceProperties2(vk.physical,&properties);
            snprintf(vk.rayStatus,sizeof(vk.rayStatus),"hardware RT ready (textured reflections/GI/flashlight shadows)");
            eprintf("*Vulkan RT: %s, max AS geometries %llu\n",vk.rayStatus,(unsigned long long)vk.accelerationProperties.maxGeometryCount);
        }
    }
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cp.queueFamilyIndex=vk.queueFamily;cp.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;vkCreateCommandPool(vk.device,&cp,nullptr,&vk.commandPool);VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=vk.commandPool;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;vkAllocateCommandBuffers(vk.device,&ca,&vk.command);VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};vkCreateSemaphore(vk.device,&si,nullptr,&vk.acquired);vkCreateSemaphore(vk.device,&si,nullptr,&vk.complete);VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};fi.flags=VK_FENCE_CREATE_SIGNALED_BIT;vkCreateFence(vk.device,&fi,nullptr,&vk.fence);
    return CreateSwapchain();
}

static void Ortho(float l,float r,float b,float t,float zn,float zf){float a=2/(r-l),bb=2/(t-b),c=1/(zn-zf),x=(l+r)/(l-r),y=(b+t)/(b-t),z=zn/(zn-zf);float m[]={a,0,0,0,0,bb,0,0,0,0,c,0,x,y,z,1};memcpy(vk.uniforms.projection,m,sizeof(m));}

static bool EnsureCaptureTexture(){
    if(!vk.captureTexture){if(vk.nextTexture>=kMaxTextures)return false;vk.captureTexture=vk.nextTexture++;}
    Texture& t=vk.textures[vk.captureTexture];
    const bool sameSize=t.alive&&t.width==vk.extent.width&&t.height==vk.extent.height;
    if(sameSize&&(!vk.postSetLayout||vk.capturePostDescriptor))return true;
    if(!sameSize){DestroyTexture(t);std::vector<uint8_t> zero((size_t)vk.extent.width*vk.extent.height*4);if(!UploadTexture(t,zero.data(),zero.size(),vk.colorFormat,vk.extent.width,vk.extent.height))return false;t.linear=false;UpdateDescriptor(vk.captureTexture);vk.captureValid=false;}
    if(vk.postSetLayout){
        if(!vk.capturePostDescriptor){VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};allocation.descriptorPool=vk.descriptorPool;allocation.descriptorSetCount=1;allocation.pSetLayouts=&vk.postSetLayout;if(!Check(vkAllocateDescriptorSets(vk.device,&allocation,&vk.capturePostDescriptor),"capture descriptor"))return false;}
        VkDescriptorImageInfo image{vk.nearestSampler,t.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};write.dstSet=vk.capturePostDescriptor;write.dstBinding=0;write.descriptorCount=1;write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;write.pImageInfo=&image;vkUpdateDescriptorSets(vk.device,1,&write,0,nullptr);
    }
    return true;
}

static void ResumeMainPass(){
    if(!vk.recording||!vk.overlayPassSuspended)return;
    VkClearValue clear[2]{};VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};pass.renderPass=vk.loadRenderPass;pass.framebuffer=vk.framebuffers[vk.imageIndex];pass.renderArea.extent=vk.extent;pass.clearValueCount=2;pass.pClearValues=clear;vkCmdBeginRenderPass(vk.command,&pass,VK_SUBPASS_CONTENTS_INLINE);vk.overlayPassSuspended=false;
}

static void SuspendMainPass(){
    if(vk.recording&&!vk.overlayPassSuspended&&!vk.inOffscreen&&!vk.inShadow){vkCmdEndRenderPass(vk.command);vk.overlayPassSuspended=true;}
}

static bool RecordTextureUpdate(Texture& texture,const void* pixels,VkDeviceSize bytes){
    if(!vk.recording||!texture.alive||!pixels)return false;SuspendMainPass();VkDeviceSize offset=(vk.overlayUploadCursor+3)&~VkDeviceSize(3);if(offset+bytes>kOverlayUploadBytes){eprinterr("Vulkan frame upload staging exhausted (%llu bytes)\n",(unsigned long long)bytes);return false;}memcpy((char*)vk.overlayUploadMap+offset,pixels,(size_t)bytes);vk.overlayUploadCursor=offset+bytes;Transition(vk.command,texture.image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkBufferImageCopy copy{};copy.bufferOffset=offset;copy.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;copy.imageSubresource.layerCount=1;copy.imageExtent={texture.width,texture.height,1};vkCmdCopyBufferToImage(vk.command,vk.overlayUpload,texture.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);Transition(vk.command,texture.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);return true;
}

static void FlushPreOverlayCopies(){
    if(!vk.recording||(!vk.framebufferStoreRequested&&!vk.captureRequested))return;
    SuspendMainPass();
    Transition(vk.command,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    if(vk.framebufferStoreRequested&&vk.framebufferTexture&&vk.textures[vk.framebufferTexture].alive){
        Texture& fb=vk.textures[vk.framebufferTexture];Transition(vk.command,fb.image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkImageBlit blit{};blit.srcSubresource.aspectMask=blit.dstSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;blit.srcSubresource.layerCount=blit.dstSubresource.layerCount=1;blit.srcOffsets[1]={(int32_t)vk.extent.width,(int32_t)vk.extent.height,1};int x=vk.framebufferRect.offset.x,y=vk.framebufferRect.offset.y,w=(int)vk.framebufferRect.extent.width,h=(int)vk.framebufferRect.extent.height;blit.dstOffsets[0]={x,y+h,0};blit.dstOffsets[1]={x+w,y,1};vkCmdBlitImage(vk.command,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,fb.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&blit,VK_FILTER_NEAREST);
        if(vk.framebufferReadback){Transition(vk.command,fb.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);VkBufferImageCopy read{};read.bufferOffset=((VkDeviceSize)y*VRAM_WIDTH+x)*4;read.bufferRowLength=VRAM_WIDTH;read.bufferImageHeight=VRAM_HEIGHT;read.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;read.imageSubresource.layerCount=1;read.imageOffset={x,y,0};read.imageExtent={(uint32_t)w,(uint32_t)h,1};vkCmdCopyImageToBuffer(vk.command,fb.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,vk.framebufferReadback,1,&read);Transition(vk.command,fb.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);QueueFramebufferReadRect(x,y,w,h);}else Transition(vk.command,fb.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);RememberDisplayFramebufferRect(x,y,w,h);
    }
    if(vk.captureRequested&&vk.captureTexture&&vk.textures[vk.captureTexture].alive){Texture& c=vk.textures[vk.captureTexture];Transition(vk.command,c.image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkImageCopy copy{};copy.srcSubresource.aspectMask=copy.dstSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;copy.srcSubresource.layerCount=copy.dstSubresource.layerCount=1;copy.extent={vk.extent.width,vk.extent.height,1};vkCmdCopyImage(vk.command,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,c.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);Transition(vk.command,c.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);vk.captureValid=true;}
    Transition(vk.command,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);vk.framebufferStoreRequested=false;vk.captureRequested=false;ResumeMainPass();
}

static bool EnsurePostTarget(){
    Texture& t=vk.postTexture;
    if(t.alive&&t.width==vk.extent.width&&t.height==vk.extent.height)return true;
    if(t.alive){vkDeviceWaitIdle(vk.device);DestroyTexture(t);}
    std::vector<uint8_t> zero((size_t)vk.extent.width*vk.extent.height*4);
    if(!UploadTexture(t,zero.data(),zero.size(),vk.colorFormat,vk.extent.width,vk.extent.height))return false;
    t.linear=true;
    if(!vk.postDescriptor){VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=vk.descriptorPool;ai.descriptorSetCount=1;ai.pSetLayouts=&vk.postSetLayout;if(!Check(vkAllocateDescriptorSets(vk.device,&ai,&vk.postDescriptor),"post descriptor"))return false;}
    VkDescriptorImageInfo image{vk.linearSampler,t.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};write.dstSet=vk.postDescriptor;write.dstBinding=0;write.descriptorCount=1;write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;write.pImageInfo=&image;vkUpdateDescriptorSets(vk.device,1,&write,0,nullptr);return true;
}

static bool EnsureShadowTarget(){
    if(vk.shadowFramebuffer)return true;
    const VkFormat format=VK_FORMAT_D32_SFLOAT;VkFormatProperties properties{};vkGetPhysicalDeviceFormatProperties(vk.physical,format,&properties);if(!(properties.optimalTilingFeatures&VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)||!(properties.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)){eprintwarn("Vulkan: sampled D32 shadow maps unsupported\n");return false;}
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};image.imageType=VK_IMAGE_TYPE_2D;image.format=format;image.extent={kShadowMapSize,kShadowMapSize,1};image.mipLevels=1;image.arrayLayers=1;image.samples=VK_SAMPLE_COUNT_1_BIT;image.tiling=VK_IMAGE_TILING_OPTIMAL;image.usage=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;if(!Check(vkCreateImage(vk.device,&image,nullptr,&vk.shadowImage),"shadow image"))return false;VkMemoryRequirements requirements{};vkGetImageMemoryRequirements(vk.device,vk.shadowImage,&requirements);VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};allocation.allocationSize=requirements.size;allocation.memoryTypeIndex=FindMemory(requirements.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);if(!Check(vkAllocateMemory(vk.device,&allocation,nullptr,&vk.shadowMemory),"shadow memory"))return false;vkBindImageMemory(vk.device,vk.shadowImage,vk.shadowMemory,0);
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};view.image=vk.shadowImage;view.viewType=VK_IMAGE_VIEW_TYPE_2D;view.format=format;view.subresourceRange.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT;view.subresourceRange.levelCount=1;view.subresourceRange.layerCount=1;if(!Check(vkCreateImageView(vk.device,&view,nullptr,&vk.shadowView),"shadow view"))return false;Immediate([&](VkCommandBuffer command){VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};barrier.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED;barrier.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;barrier.image=vk.shadowImage;barrier.subresourceRange.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT;barrier.subresourceRange.levelCount=1;barrier.subresourceRange.layerCount=1;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,0,nullptr,1,&barrier);});
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};sampler.magFilter=sampler.minFilter=VK_FILTER_NEAREST;sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;sampler.borderColor=VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;sampler.maxLod=1.0f;if(!Check(vkCreateSampler(vk.device,&sampler,nullptr,&vk.shadowSampler),"shadow sampler"))return false;
    VkAttachmentDescription attachment{};attachment.format=format;attachment.samples=VK_SAMPLE_COUNT_1_BIT;attachment.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;attachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;attachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;attachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;attachment.initialLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;attachment.finalLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;VkAttachmentReference depth{0,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};VkSubpassDescription subpass{};subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;subpass.pDepthStencilAttachment=&depth;VkSubpassDependency dependencies[2]{};dependencies[0].srcSubpass=VK_SUBPASS_EXTERNAL;dependencies[0].dstSubpass=0;dependencies[0].srcStageMask=VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;dependencies[0].dstStageMask=VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;dependencies[0].srcAccessMask=VK_ACCESS_SHADER_READ_BIT;dependencies[0].dstAccessMask=VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;dependencies[0].dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT;dependencies[1].srcSubpass=0;dependencies[1].dstSubpass=VK_SUBPASS_EXTERNAL;dependencies[1].srcStageMask=VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;dependencies[1].dstStageMask=VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;dependencies[1].srcAccessMask=VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;dependencies[1].dstAccessMask=VK_ACCESS_SHADER_READ_BIT;dependencies[1].dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT;VkRenderPassCreateInfo renderPass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};renderPass.attachmentCount=1;renderPass.pAttachments=&attachment;renderPass.subpassCount=1;renderPass.pSubpasses=&subpass;renderPass.dependencyCount=2;renderPass.pDependencies=dependencies;if(!Check(vkCreateRenderPass(vk.device,&renderPass,nullptr,&vk.shadowRenderPass),"shadow render pass"))return false;
    VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};framebuffer.renderPass=vk.shadowRenderPass;framebuffer.attachmentCount=1;framebuffer.pAttachments=&vk.shadowView;framebuffer.width=kShadowMapSize;framebuffer.height=kShadowMapSize;framebuffer.layers=1;if(!Check(vkCreateFramebuffer(vk.device,&framebuffer,nullptr,&vk.shadowFramebuffer),"shadow framebuffer"))return false;
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};stage.stage=VK_SHADER_STAGE_VERTEX_BIT;stage.module=vk.shadowVert;stage.pName="main";VkVertexInputBindingDescription binding{0,sizeof(GrVertex),VK_VERTEX_INPUT_RATE_VERTEX};VkVertexInputAttributeDescription attributes[2]={{7,0,VK_FORMAT_R32G32B32_SFLOAT,36},{8,0,VK_FORMAT_R32G32B32_SFLOAT,48}};VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};vertexInput.vertexBindingDescriptionCount=1;vertexInput.pVertexBindingDescriptions=&binding;vertexInput.vertexAttributeDescriptionCount=2;vertexInput.pVertexAttributeDescriptions=attributes;VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};viewport.viewportCount=1;viewport.scissorCount=1;VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};raster.polygonMode=VK_POLYGON_MODE_FILL;raster.cullMode=VK_CULL_MODE_NONE;raster.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;raster.depthBiasEnable=VK_TRUE;raster.lineWidth=1.0f;VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};multisample.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;VkPipelineDepthStencilStateCreateInfo depthState{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};depthState.depthTestEnable=VK_TRUE;depthState.depthWriteEnable=VK_TRUE;depthState.depthCompareOp=VK_COMPARE_OP_LESS;VkDynamicState states[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR,VK_DYNAMIC_STATE_DEPTH_BIAS};VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dynamic.dynamicStateCount=3;dynamic.pDynamicStates=states;VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};pipeline.stageCount=1;pipeline.pStages=&stage;pipeline.pVertexInputState=&vertexInput;pipeline.pInputAssemblyState=&assembly;pipeline.pViewportState=&viewport;pipeline.pRasterizationState=&raster;pipeline.pMultisampleState=&multisample;pipeline.pDepthStencilState=&depthState;pipeline.pDynamicState=&dynamic;pipeline.layout=vk.pipelineLayout;pipeline.renderPass=vk.shadowRenderPass;return Check(vkCreateGraphicsPipelines(vk.device,VK_NULL_HANDLE,1,&pipeline,nullptr,&vk.shadowPipeline),"shadow pipeline");
}

static bool EnsureOffscreenTarget(){
    if(vk.offscreenFramebuffer)return true;if(!vk.offscreenTexture){if(vk.nextTexture>=kMaxTextures)return false;vk.offscreenTexture=vk.nextTexture++;}
    Texture& t=vk.textures[vk.offscreenTexture];std::vector<uint8_t> zero((size_t)VRAM_WIDTH*VRAM_HEIGHT*4);
    if(!UploadTexture(t,zero.data(),zero.size(),vk.colorFormat,VRAM_WIDTH,VRAM_HEIGHT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT))return false;t.linear=false;UpdateDescriptor(vk.offscreenTexture);
    if(!CreateCompatibleRenderPass(VK_ATTACHMENT_LOAD_OP_LOAD,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_ATTACHMENT_LOAD_OP_CLEAR,&vk.offscreenRenderPass))return false;
    VkImageCreateInfo di{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};di.imageType=VK_IMAGE_TYPE_2D;di.format=vk.depthFormat;di.extent={VRAM_WIDTH,VRAM_HEIGHT,1};di.mipLevels=1;di.arrayLayers=1;di.samples=VK_SAMPLE_COUNT_1_BIT;di.tiling=VK_IMAGE_TILING_OPTIMAL;di.usage=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if(!Check(vkCreateImage(vk.device,&di,nullptr,&vk.offscreenDepthImage),"offscreen depth"))return false;VkMemoryRequirements mr{};vkGetImageMemoryRequirements(vk.device,vk.offscreenDepthImage,&mr);VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=mr.size;ai.memoryTypeIndex=FindMemory(mr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);if(!Check(vkAllocateMemory(vk.device,&ai,nullptr,&vk.offscreenDepthMemory),"offscreen depth memory"))return false;vkBindImageMemory(vk.device,vk.offscreenDepthImage,vk.offscreenDepthMemory,0);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};vi.image=vk.offscreenDepthImage;vi.viewType=VK_IMAGE_VIEW_TYPE_2D;vi.format=vk.depthFormat;vi.subresourceRange.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT;vi.subresourceRange.levelCount=1;vi.subresourceRange.layerCount=1;if(!Check(vkCreateImageView(vk.device,&vi,nullptr,&vk.offscreenDepthView),"offscreen depth view"))return false;
    VkImageView views[]={t.view,vk.offscreenDepthView};VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};fi.renderPass=vk.offscreenRenderPass;fi.attachmentCount=2;fi.pAttachments=views;fi.width=VRAM_WIDTH;fi.height=VRAM_HEIGHT;fi.layers=1;return Check(vkCreateFramebuffer(vk.device,&fi,nullptr,&vk.offscreenFramebuffer),"offscreen framebuffer");
}

static void RecordVRAMUpload(VkCommandBuffer cmd){
    if(!vk.vramStagingMap||!g_vramTexture||!vk.textures[g_vramTexture].alive)return;
    memcpy(vk.vramStagingMap,vram,sizeof(vram));Texture& t=vk.textures[g_vramTexture];
    Transition(cmd,t.image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy copy{};copy.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;copy.imageSubresource.layerCount=1;copy.imageExtent={VRAM_WIDTH,VRAM_HEIGHT,1};
    vkCmdCopyBufferToImage(cmd,vk.vramStaging,t.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
    Transition(cmd,t.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
}

int GR_RayTracingAvailable(void){return vk.rayQueryAvailable?1:0;}
int GR_RayTracingEnabled(void){return (g_cfg_rtgi&&vk.rayQueryAvailable&&vk.rtProbeAddress&&vk.rtCurrent.tlas)?1:0;}
const char* GR_RayTracingStatus(void){return vk.rayStatus;}

int GR_InitialiseRender(char* name,int width,int height,int fullscreen){uint32_t flags=SDL_WINDOW_VULKAN|SDL_WINDOW_RESIZABLE;if(fullscreen==1)flags|=SDL_WINDOW_FULLSCREEN;if(fullscreen==2)flags|=SDL_WINDOW_FULLSCREEN_DESKTOP;g_window=SDL_CreateWindow(name,SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,width,height,flags);if(!g_window){eprinterr("SDL window: %s\n",SDL_GetError());return 0;}if(!InitDevice())return 0;eprintf("*Renderer: Vulkan\n");return 1;}

int GR_InitialisePSX(){memset(vram,0,sizeof(vram));VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(vk.physical,&p);vk.uniformStride=(sizeof(Uniforms)+p.limits.minUniformBufferOffsetAlignment-1)&~(p.limits.minUniformBufferOffsetAlignment-1);VertexSlice firstVertexSlice{};CreateBuffer(kVertexBytes,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,firstVertexSlice.buffer,firstVertexSlice.memory,&firstVertexSlice.mapped);vk.vertexSlices.push_back(firstVertexSlice);CreateBuffer(vk.uniformStride*kUniformSlots,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,vk.uniformBuffer,vk.uniformMemory,&vk.uniformMap);
    /* Exercise the exact allocation path future BLAS/TLAS inputs will use. A
     * non-zero address proves both device-address memory and the driver entry
     * point work, instead of reporting support from extension strings alone. */
    if(vk.rayQueryAvailable){
        if(CreateBuffer(256,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,vk.rtProbeBuffer,vk.rtProbeMemory,nullptr,true)){
            VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};addressInfo.buffer=vk.rtProbeBuffer;vk.rtProbeAddress=vk.getBufferDeviceAddress(vk.device,&addressInfo);
        }
        if(!vk.rtProbeAddress){
            vk.rayQueryAvailable=false;snprintf(vk.rayStatus,sizeof(vk.rayStatus),"unavailable (device-address probe failed)");
            if(vk.rtProbeBuffer){vkDestroyBuffer(vk.device,vk.rtProbeBuffer,nullptr);vk.rtProbeBuffer=VK_NULL_HANDLE;}
            if(vk.rtProbeMemory){vkFreeMemory(vk.device,vk.rtProbeMemory,nullptr);vk.rtProbeMemory=VK_NULL_HANDLE;}
        }else eprintf("*Vulkan RT: device-address probe 0x%llx\n",(unsigned long long)vk.rtProbeAddress);
    }
    VkDescriptorSetLayoutBinding bindings[4]={{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},{1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},{2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},{3,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=4;li.pBindings=bindings;vkCreateDescriptorSetLayout(vk.device,&li,nullptr,&vk.setLayout);
    if(vk.rayQueryAvailable){VkDescriptorSetLayoutBinding rtBindings[2]={{0,VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};VkDescriptorSetLayoutCreateInfo rtLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};rtLayoutInfo.bindingCount=2;rtLayoutInfo.pBindings=rtBindings;if(!Check(vkCreateDescriptorSetLayout(vk.device,&rtLayoutInfo,nullptr,&vk.rtSetLayout),"RT descriptor layout"))vk.rayQueryAvailable=false;}
    VkDescriptorPoolSize sizes[4]={{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,kMaxTextures},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,kMaxTextures*3+8},{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1}};VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=kMaxTextures+9;dpi.poolSizeCount=vk.rayQueryAvailable?4u:2u;dpi.pPoolSizes=sizes;vkCreateDescriptorPool(vk.device,&dpi,nullptr,&vk.descriptorPool);
    VkPushConstantRange overlayPush{VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(OverlayConstants)};VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&vk.setLayout;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&overlayPush;vkCreatePipelineLayout(vk.device,&pli,nullptr,&vk.pipelineLayout);
    if(vk.rayQueryAvailable&&vk.rtSetLayout){VkDescriptorSetLayout rtLayouts[]={vk.setLayout,vk.rtSetLayout};VkPipelineLayoutCreateInfo rtPipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};rtPipelineInfo.setLayoutCount=2;rtPipelineInfo.pSetLayouts=rtLayouts;if(!Check(vkCreatePipelineLayout(vk.device,&rtPipelineInfo,nullptr,&vk.rtPipelineLayout),"RT pipeline layout"))vk.rayQueryAvailable=false;else{VkDescriptorSetAllocateInfo rtAllocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};rtAllocate.descriptorPool=vk.descriptorPool;rtAllocate.descriptorSetCount=1;rtAllocate.pSetLayouts=&vk.rtSetLayout;if(!Check(vkAllocateDescriptorSets(vk.device,&rtAllocate,&vk.rtDescriptor),"RT descriptor"))vk.rayQueryAvailable=false;}}
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};sci.magFilter=VK_FILTER_NEAREST;sci.minFilter=VK_FILTER_NEAREST;sci.addressModeU=sci.addressModeV=sci.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;sci.maxLod=1;vkCreateSampler(vk.device,&sci,nullptr,&vk.nearestSampler);sci.magFilter=sci.minFilter=VK_FILTER_LINEAR;vkCreateSampler(vk.device,&sci,nullptr,&vk.linearSampler);vk.vert=Shader(g_psx_vert_spv,sizeof(g_psx_vert_spv));vk.frag=Shader(g_psx_frag_spv,sizeof(g_psx_frag_spv));if(vk.rayQueryAvailable)vk.rtFrag=Shader(g_psx_rt_frag_spv,sizeof(g_psx_rt_frag_spv));vk.shadowVert=Shader(g_shadow_vert_spv,sizeof(g_shadow_vert_spv));vk.overlayVert=Shader(g_overlay_vert_spv,sizeof(g_overlay_vert_spv));vk.overlayFrag=Shader(g_overlay_frag_spv,sizeof(g_overlay_frag_spv));vk.lineVert=Shader(g_line_vert_spv,sizeof(g_line_vert_spv));vk.lineFrag=Shader(g_line_frag_spv,sizeof(g_line_frag_spv));CreateBuffer(kOverlayUploadBytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,vk.overlayUpload,vk.overlayUploadMemory,&vk.overlayUploadMap);CreateBuffer(kOverlayLineBytes,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,vk.overlayLines,vk.overlayLineMemory,&vk.overlayLineMap);CreateOverlayPipelines();EnsureShadowTarget();
    g_vramTexture=vk.nextTexture++;UploadTexture(vk.textures[g_vramTexture],vram,sizeof(vram),VK_FORMAT_R8G8_UNORM,VRAM_WIDTH,VRAM_HEIGHT);UpdateDescriptor(g_vramTexture);uint32_t white=0xffffffff;g_whiteTexture=vk.nextTexture++;UploadTexture(vk.textures[g_whiteTexture],&white,4,VK_FORMAT_R8G8B8A8_UNORM,1,1);UpdateDescriptor(g_whiteTexture);vk.framebufferTexture=vk.nextTexture++;std::vector<uint8_t> fbZero((size_t)VRAM_WIDTH*VRAM_HEIGHT*4);UploadTexture(vk.textures[vk.framebufferTexture],fbZero.data(),fbZero.size(),VK_FORMAT_R8G8B8A8_UNORM,VRAM_WIDTH,VRAM_HEIGHT);UpdateDescriptor(vk.framebufferTexture);EnsureOffscreenTarget();UpdateDescriptor(g_vramTexture);UpdateDescriptor(g_whiteTexture);CreateBuffer(sizeof(vram),VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,vk.vramStaging,vk.vramStagingMemory,&vk.vramStagingMap);CreateBuffer((VkDeviceSize)VRAM_WIDTH*VRAM_HEIGHT*4,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,vk.framebufferReadback,vk.framebufferReadbackMemory,&vk.framebufferReadbackMap);vk.boundTexture=g_vramTexture;Ortho(0,320,240,0,-1,1);GR_InitPostProcess();return 1;}

void GR_BeginScene(){
    if(vk.recreate){DestroySwapchain();if(!CreateSwapchain())return;vk.recreate=false;}
    if(!vk.swapchain)return;
    EnsureCaptureTexture();if(vk.postSetLayout)EnsurePostTarget();
    vkWaitForFences(vk.device,1,&vk.fence,VK_TRUE,UINT64_MAX);DestroyRetiredTextures();PromoteRayScene();vk.rtFramePositions.clear();vk.rtFrameMaterials.clear();vk.currentVertexSlice=nullptr;
    /* Headless/automated validation hook: seed one harmless view-space triangle
     * so device creation, BLAS/TLAS build, descriptor binding and the RT shader
     * can all be exercised without navigating from the title into gameplay. */
    static int rtSmoke=-1;if(rtSmoke<0)rtSmoke=getenv("PSYX_RT_SMOKE")?1:0;if(rtSmoke&&g_cfg_rtgi&&vk.rayQueryAvailable){const float triangle[]={-64.0f,-64.0f,512.0f,64.0f,-64.0f,512.0f,0.0f,64.0f,512.0f};vk.rtFramePositions.assign(triangle,triangle+9);RayMaterial material{};material.color0[0]=material.color0[1]=material.color0[2]=material.color0[3]=1.0f;memcpy(material.color1,material.color0,sizeof(material.color0));memcpy(material.color2,material.color0,sizeof(material.color0));vk.rtFrameMaterials.push_back(material);}
    for(uint32_t i=0;i<vk.framebufferReadPendingCount&&vk.framebufferReadReadyCount<kFramebufferRegions;i++)vk.framebufferReadReadyRects[vk.framebufferReadReadyCount++]=vk.framebufferReadPendingRects[i];
    vk.framebufferReadPendingCount=0;vk.overlayUploadCursor=0;vk.overlayPassSuspended=false;vk.vertexSliceIndex=0;
    VkResult ar=vkAcquireNextImageKHR(vk.device,vk.swapchain,UINT64_MAX,vk.acquired,VK_NULL_HANDLE,&vk.imageIndex);
    if(ar==VK_ERROR_OUT_OF_DATE_KHR){vk.recreate=true;return;}
    if(ar!=VK_SUCCESS&&ar!=VK_SUBOPTIMAL_KHR){eprinterr("Vulkan: acquire failed (%d)\n",(int)ar);return;}
    if(ar==VK_SUBOPTIMAL_KHR)vk.recreate=true;
    vkResetFences(vk.device,1,&vk.fence);
    vkResetCommandBuffer(vk.command,0);VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};vkBeginCommandBuffer(vk.command,&bi);RecordVRAMUpload(vk.command);
    VkClearValue cv[2]{};cv[0].color={{0,0,0,1}};cv[1].depthStencil={1,0};VkRenderPassBeginInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};ri.renderPass=vk.renderPass;ri.framebuffer=vk.framebuffers[vk.imageIndex];ri.renderArea.extent=vk.extent;ri.clearValueCount=2;ri.pClearValues=cv;vkCmdBeginRenderPass(vk.command,&ri,VK_SUBPASS_CONTENTS_INLINE);vk.uniformSlot=0;vk.recording=true;vk.inOffscreen=false;GR_SetViewPort(0,0,g_windowWidth,g_windowHeight);
    vk.wire=g_dbg_wireframeMode!=0&&vk.fillModeNonSolid;
    if(vk.wire){VkClearAttachment attachment{};attachment.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;attachment.clearValue.color={{0.1f,0.1f,0.1f,1.0f}};VkClearRect rect{};rect.rect.extent=vk.extent;rect.layerCount=1;vkCmdClearAttachments(vk.command,1,&attachment,1,&rect);}
}
void GR_EndScene(){vk.wire=false;}
void GR_SwapWindow(){
    if(!vk.recording)return;
    if(vk.inOffscreen)GR_SetOffscreenState(&vk.offscreenRect,0);
    ResumeMainPass();
    /* Validation-only smoke hook: exercise the depth-only pass and main-pass
     * restoration without stealing desktop focus to navigate into gameplay. */
    static int shadowTestPending=-1;
    if(shadowTestPending<0)shadowTestPending=getenv("PSYX_SHADOW_TEST")?1:0;
    if(shadowTestPending){GR_ShadowPassBegin();GR_ShadowPassEnd();shadowTestPending=0;}
    vkCmdEndRenderPass(vk.command);
    bool transferSource=vk.framebufferStoreRequested||vk.captureRequested||vk.postRequested;
    if(transferSource)Transition(vk.command,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    if(vk.framebufferStoreRequested&&vk.framebufferTexture&&vk.textures[vk.framebufferTexture].alive){
        Texture& fb=vk.textures[vk.framebufferTexture];Transition(vk.command,fb.image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageBlit blit{};blit.srcSubresource.aspectMask=blit.dstSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;blit.srcSubresource.layerCount=blit.dstSubresource.layerCount=1;blit.srcOffsets[1]={(int32_t)vk.extent.width,(int32_t)vk.extent.height,1};
        int x=vk.framebufferRect.offset.x,y=vk.framebufferRect.offset.y,w=(int)vk.framebufferRect.extent.width,h=(int)vk.framebufferRect.extent.height;blit.dstOffsets[0]={x,y+h,0};blit.dstOffsets[1]={x+w,y,1};
        vkCmdBlitImage(vk.command,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,fb.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&blit,VK_FILTER_NEAREST);
        if(vk.framebufferReadback){Transition(vk.command,fb.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);VkBufferImageCopy read{};read.bufferOffset=((VkDeviceSize)y*VRAM_WIDTH+x)*4;read.bufferRowLength=VRAM_WIDTH;read.bufferImageHeight=VRAM_HEIGHT;read.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;read.imageSubresource.layerCount=1;read.imageOffset={x,y,0};read.imageExtent={(uint32_t)w,(uint32_t)h,1};vkCmdCopyImageToBuffer(vk.command,fb.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,vk.framebufferReadback,1,&read);Transition(vk.command,fb.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);QueueFramebufferReadRect(x,y,w,h);}else Transition(vk.command,fb.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);RememberDisplayFramebufferRect(x,y,w,h);
    }
    if(vk.captureRequested&&vk.captureTexture&&vk.textures[vk.captureTexture].alive){Texture& c=vk.textures[vk.captureTexture];Transition(vk.command,c.image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkImageCopy copy{};copy.srcSubresource.aspectMask=copy.dstSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;copy.srcSubresource.layerCount=copy.dstSubresource.layerCount=1;copy.extent={vk.extent.width,vk.extent.height,1};vkCmdCopyImage(vk.command,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,c.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);Transition(vk.command,c.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);vk.captureValid=true;}
    if(vk.postRequested&&vk.postTexture.alive&&vk.postPipeline&&vk.postDescriptor){
        Transition(vk.command,vk.postTexture.image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageCopy copy{};copy.srcSubresource.aspectMask=copy.dstSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;copy.srcSubresource.layerCount=copy.dstSubresource.layerCount=1;copy.extent={vk.extent.width,vk.extent.height,1};
        vkCmdCopyImage(vk.command,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,vk.postTexture.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
        Transition(vk.command,vk.postTexture.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(vk.command,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        VkClearValue clear[2]{};VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};pass.renderPass=vk.loadRenderPass;pass.framebuffer=vk.framebuffers[vk.imageIndex];pass.renderArea.extent=vk.extent;pass.clearValueCount=2;pass.pClearValues=clear;vkCmdBeginRenderPass(vk.command,&pass,VK_SUBPASS_CONTENTS_INLINE);
        VkViewport viewport{0.0f,0.0f,(float)vk.extent.width,(float)vk.extent.height,0.0f,1.0f};VkRect2D scissor{{0,0},vk.extent};vkCmdSetViewport(vk.command,0,1,&viewport);vkCmdSetScissor(vk.command,0,1,&scissor);vkCmdBindPipeline(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,vk.postPipeline);vkCmdBindDescriptorSets(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,vk.postPipelineLayout,0,1,&vk.postDescriptor,0,nullptr);
        PostConstants pc{};pc.mode=std::clamp(g_cfg_postProcess,0,8);pc.tonemapMode=std::clamp(g_cfg_tonemap,0,3);pc.texelSize[0]=1.0f/std::max(1u,vk.extent.width);pc.texelSize[1]=1.0f/std::max(1u,vk.extent.height);pc.time=(float)(vk.postFrame++&1023u);pc.postIntensity=std::clamp(g_cfg_postProcessIntensity,0.0f,1.0f);pc.tonemapIntensity=std::clamp(g_cfg_tonemapIntensity,0.0f,1.0f);vkCmdPushConstants(vk.command,vk.postPipelineLayout,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(pc),&pc);vkCmdDraw(vk.command,3,1,0,0);vkCmdEndRenderPass(vk.command);
    }else if(transferSource)Transition(vk.command,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    vk.framebufferStoreRequested=false;vk.captureRequested=false;vk.postRequested=false;
    BuildRaySceneForNextFrame(vk.command);vkEndCommandBuffer(vk.command);VkPipelineStageFlags wait=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.waitSemaphoreCount=1;si.pWaitSemaphores=&vk.acquired;si.pWaitDstStageMask=&wait;si.commandBufferCount=1;si.pCommandBuffers=&vk.command;si.signalSemaphoreCount=1;si.pSignalSemaphores=&vk.complete;vkQueueSubmit(vk.queue,1,&si,vk.fence);VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};pi.waitSemaphoreCount=1;pi.pWaitSemaphores=&vk.complete;pi.swapchainCount=1;pi.pSwapchains=&vk.swapchain;pi.pImageIndices=&vk.imageIndex;VkResult r=vkQueuePresentKHR(vk.queue,&pi);if(r==VK_ERROR_OUT_OF_DATE_KHR||r==VK_SUBOPTIMAL_KHR)vk.recreate=true;vk.recording=false;
}

void GR_Shutdown(){
    if(!vk.device)return;vkDeviceWaitIdle(vk.device);DestroyRayScene(vk.rtCurrent);DestroyRayScene(vk.rtPending);if(vk.shadowPipeline)vkDestroyPipeline(vk.device,vk.shadowPipeline,nullptr);if(vk.shadowFramebuffer)vkDestroyFramebuffer(vk.device,vk.shadowFramebuffer,nullptr);if(vk.shadowRenderPass)vkDestroyRenderPass(vk.device,vk.shadowRenderPass,nullptr);if(vk.shadowSampler)vkDestroySampler(vk.device,vk.shadowSampler,nullptr);if(vk.shadowView)vkDestroyImageView(vk.device,vk.shadowView,nullptr);if(vk.shadowImage)vkDestroyImage(vk.device,vk.shadowImage,nullptr);if(vk.shadowMemory)vkFreeMemory(vk.device,vk.shadowMemory,nullptr);if(vk.offscreenFramebuffer)vkDestroyFramebuffer(vk.device,vk.offscreenFramebuffer,nullptr);if(vk.offscreenRenderPass)vkDestroyRenderPass(vk.device,vk.offscreenRenderPass,nullptr);if(vk.offscreenDepthView)vkDestroyImageView(vk.device,vk.offscreenDepthView,nullptr);if(vk.offscreenDepthImage)vkDestroyImage(vk.device,vk.offscreenDepthImage,nullptr);if(vk.offscreenDepthMemory)vkFreeMemory(vk.device,vk.offscreenDepthMemory,nullptr);DestroyRetiredTextures();DestroyTexture(vk.postTexture);for(auto& t:vk.textures)DestroyTexture(t);DestroySwapchain();
    if(vk.vert)vkDestroyShaderModule(vk.device,vk.vert,nullptr);if(vk.frag)vkDestroyShaderModule(vk.device,vk.frag,nullptr);if(vk.rtFrag)vkDestroyShaderModule(vk.device,vk.rtFrag,nullptr);if(vk.postVert)vkDestroyShaderModule(vk.device,vk.postVert,nullptr);if(vk.postFrag)vkDestroyShaderModule(vk.device,vk.postFrag,nullptr);if(vk.shadowVert)vkDestroyShaderModule(vk.device,vk.shadowVert,nullptr);if(vk.overlayVert)vkDestroyShaderModule(vk.device,vk.overlayVert,nullptr);if(vk.overlayFrag)vkDestroyShaderModule(vk.device,vk.overlayFrag,nullptr);if(vk.lineVert)vkDestroyShaderModule(vk.device,vk.lineVert,nullptr);if(vk.lineFrag)vkDestroyShaderModule(vk.device,vk.lineFrag,nullptr);
    if(vk.nearestSampler)vkDestroySampler(vk.device,vk.nearestSampler,nullptr);if(vk.linearSampler)vkDestroySampler(vk.device,vk.linearSampler,nullptr);
    if(vk.pipelineLayout)vkDestroyPipelineLayout(vk.device,vk.pipelineLayout,nullptr);if(vk.rtPipelineLayout)vkDestroyPipelineLayout(vk.device,vk.rtPipelineLayout,nullptr);if(vk.postPipelineLayout)vkDestroyPipelineLayout(vk.device,vk.postPipelineLayout,nullptr);if(vk.descriptorPool)vkDestroyDescriptorPool(vk.device,vk.descriptorPool,nullptr);if(vk.setLayout)vkDestroyDescriptorSetLayout(vk.device,vk.setLayout,nullptr);if(vk.rtSetLayout)vkDestroyDescriptorSetLayout(vk.device,vk.rtSetLayout,nullptr);if(vk.postSetLayout)vkDestroyDescriptorSetLayout(vk.device,vk.postSetLayout,nullptr);
    for(VertexSlice& slice:vk.vertexSlices){if(slice.mapped)vkUnmapMemory(vk.device,slice.memory);if(slice.buffer)vkDestroyBuffer(vk.device,slice.buffer,nullptr);if(slice.memory)vkFreeMemory(vk.device,slice.memory,nullptr);}vk.vertexSlices.clear();
    if(vk.uniformMap)vkUnmapMemory(vk.device,vk.uniformMemory);if(vk.uniformBuffer)vkDestroyBuffer(vk.device,vk.uniformBuffer,nullptr);if(vk.uniformMemory)vkFreeMemory(vk.device,vk.uniformMemory,nullptr);
    if(vk.vramStagingMap)vkUnmapMemory(vk.device,vk.vramStagingMemory);if(vk.vramStaging)vkDestroyBuffer(vk.device,vk.vramStaging,nullptr);if(vk.vramStagingMemory)vkFreeMemory(vk.device,vk.vramStagingMemory,nullptr);
    if(vk.framebufferReadbackMap)vkUnmapMemory(vk.device,vk.framebufferReadbackMemory);if(vk.framebufferReadback)vkDestroyBuffer(vk.device,vk.framebufferReadback,nullptr);if(vk.framebufferReadbackMemory)vkFreeMemory(vk.device,vk.framebufferReadbackMemory,nullptr);
    if(vk.overlayUploadMap)vkUnmapMemory(vk.device,vk.overlayUploadMemory);if(vk.overlayUpload)vkDestroyBuffer(vk.device,vk.overlayUpload,nullptr);if(vk.overlayUploadMemory)vkFreeMemory(vk.device,vk.overlayUploadMemory,nullptr);
    if(vk.overlayLineMap)vkUnmapMemory(vk.device,vk.overlayLineMemory);if(vk.overlayLines)vkDestroyBuffer(vk.device,vk.overlayLines,nullptr);if(vk.overlayLineMemory)vkFreeMemory(vk.device,vk.overlayLineMemory,nullptr);
    if(vk.rtProbeBuffer)vkDestroyBuffer(vk.device,vk.rtProbeBuffer,nullptr);if(vk.rtProbeMemory)vkFreeMemory(vk.device,vk.rtProbeMemory,nullptr);
    vkDestroySemaphore(vk.device,vk.acquired,nullptr);vkDestroySemaphore(vk.device,vk.complete,nullptr);vkDestroyFence(vk.device,vk.fence,nullptr);vkDestroyCommandPool(vk.device,vk.commandPool,nullptr);vkDestroyDevice(vk.device,nullptr);vkDestroySurfaceKHR(vk.instance,vk.surface,nullptr);vkDestroyInstance(vk.instance,nullptr);vk=VulkanState{};
}
void GR_ResetDevice(){vk.recreate=true;} void GR_UpdateSwapIntervalState(int interval){int normalized=interval>0?1:0;if(vk.swapInterval!=normalized){vk.swapInterval=normalized;vk.recreate=true;}}
void GR_WaitIdle(){if(vk.device)vkDeviceWaitIdle(vk.device);}

void GR_Ortho2D(float l,float r,float b,float t,float zn,float zf){Ortho(l,r,b,t,zn,zf);}
void GR_Perspective3D(float fov,float width,float height,float zn,float zf){float h=cosf(fov*.5f)/sinf(fov*.5f),w=h*height/width;float m[]={w,0,0,0,0,h,0,0,0,0,zf/(zf-zn),1,0,0,-zf*zn/(zf-zn),0};memcpy(vk.uniforms.projection3D,m,sizeof(m));}
void GR_SetViewPort(int x,int y,int w,int h){vk.viewport={(float)x,(float)(y+h),(float)w,(float)-h,0,1};}
void GR_SetScissorState(int e){vk.scissorEnabled=e!=0;}
void GR_SetupClipMode(const RECT16* r,int e){
    if(!r){GR_SetScissorState(0);return;}
    bool on=e&&(activeDispEnv.isinter||r->x-activeDispEnv.disp.x>0||r->y-activeDispEnv.disp.y>0||r->w<activeDispEnv.disp.w-1||r->h<activeDispEnv.disp.h-1);
    GR_SetScissorState(on);if(!on)return;
    float dw=(float)std::max(1,(int)activeDispEnv.disp.w),dh=(float)std::max(1,(int)activeDispEnv.disp.h);
    float nx=(r->x-activeDispEnv.disp.x)/dw,ny=(r->y-activeDispEnv.disp.y)/dh,nw=r->w/dw,nh=r->h/dh;
    const int targetW=vk.inOffscreen?std::max(1,(int)vk.offscreenRect.w):g_windowWidth;
    const int targetH=vk.inOffscreen?std::max(1,(int)vk.offscreenRect.h):g_windowHeight;
    int x=(int)lroundf(nx*targetW),y=(int)lroundf(ny*targetH),w=(int)lroundf(nw*targetW),h=(int)lroundf(nh*targetH);
    x=std::clamp(x,0,targetW);y=std::clamp(y,0,targetH);w=std::clamp(w,0,targetW-x);h=std::clamp(h,0,targetH-y);
    vk.scissor.offset={x,y};vk.scissor.extent={(uint32_t)w,(uint32_t)h};
}
extern "C" int PsyX_MapWindowToViewport(int mx,int my,float* ox,float* oy){int x=0,w=g_windowWidth;bool pillar=(g_PcHorPlusEnabled&&g_PcWidescreenMode==0)||(!g_PcHorPlusEnabled&&g_PcMenuPillarbox);if(pillar&&g_windowHeight>0&&(float)g_windowWidth/g_windowHeight>4.f/3.f){w=(int)(g_windowHeight*4.f/3.f+.5f);x=(g_windowWidth-w)/2;}float fx=w?(float)(mx-x)/w:0,fy=g_windowHeight?(float)my/g_windowHeight:0;if(ox)*ox=fx;if(oy)*oy=fy;return fx>=0&&fx<=1&&fy>=0&&fy<=1;}
void GR_SetOffscreenState(const RECT16* r,int e){
    if(!r)return;
    if(e){Ortho(0,(float)r->w,(float)r->h,0,-1,1);GR_SetViewPort(0,0,r->w,r->h);}else{float W=(float)activeDispEnv.disp.w,H=(float)activeDispEnv.disp.h,top=0,bot=g_PcHorPlusEnabled?H*(g_PsxUIOrthoPass?1:g_PsxWorldVScale):H;float aspect=g_windowHeight?(float)g_windowWidth/g_windowHeight:4.f/3.f,scale=aspect/(W/H);if(g_PcHorPlusEnabled&&g_PcWidescreenMode==1&&scale>1){float cx=W*.5f,half=(W*.5f*scale*g_PsxPixelAspect)/(g_PsxUIOrthoPass?1:g_PsxWorldHScale);Ortho(cx-half,cx+half,bot,top,-1,1);}else Ortho(0,W,bot,top,-1,1);int x=0,w=g_windowWidth;bool pillar=(g_PcHorPlusEnabled&&g_PcWidescreenMode==0)||(!g_PcHorPlusEnabled&&g_PcMenuPillarbox);if(pillar&&g_windowHeight&&aspect>4.f/3.f){w=(int)(g_windowHeight*4.f/3.f+.5f);x=(g_windowWidth-w)/2;}GR_SetViewPort(x,0,w,g_windowHeight);}
    if(!vk.recording||e==vk.inOffscreen)return;
    {static int logCount=0;if(logCount<16){eprintf("[VK/OFFSCREEN] %s rect=(%d,%d %dx%d)\n",e?"begin":"end",r->x,r->y,r->w,r->h);logCount++;}}
    vkCmdEndRenderPass(vk.command);
    if(e){
        vk.offscreenRect=*r;vk.offscreenRect.w=(short)std::clamp((int)vk.offscreenRect.w,1,VRAM_WIDTH);vk.offscreenRect.h=(short)std::clamp((int)vk.offscreenRect.h,1,VRAM_HEIGHT);
        VkClearValue clear[2]{};clear[0].color={{0.5f,0.5f,0.5f,0.0f}};clear[1].depthStencil={1,0};VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};bi.renderPass=vk.offscreenRenderPass;bi.framebuffer=vk.offscreenFramebuffer;bi.renderArea.extent={(uint32_t)vk.offscreenRect.w,(uint32_t)vk.offscreenRect.h};bi.clearValueCount=2;bi.pClearValues=clear;vkCmdBeginRenderPass(vk.command,&bi,VK_SUBPASS_CONTENTS_INLINE);VkClearAttachment a[2]{};a[0].aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;a[0].clearValue=clear[0];a[1].aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT;a[1].clearValue=clear[1];VkClearRect cr{};cr.rect.extent=bi.renderArea.extent;cr.layerCount=1;vkCmdClearAttachments(vk.command,2,a,1,&cr);vk.inOffscreen=true;
    }else{
        if(!g_PsxSkipFramebufferStore&&vk.framebufferTexture&&vk.offscreenTexture){Texture& src=vk.textures[vk.offscreenTexture];Texture& dst=vk.textures[vk.framebufferTexture];Transition(vk.command,src.image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);Transition(vk.command,dst.image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);int x=std::clamp((int)vk.offscreenRect.x,0,VRAM_WIDTH),y=std::clamp((int)vk.offscreenRect.y,0,VRAM_HEIGHT),w=std::clamp((int)vk.offscreenRect.w,0,VRAM_WIDTH-x),h=std::clamp((int)vk.offscreenRect.h,0,VRAM_HEIGHT-y);VkImageBlit b{};b.srcSubresource.aspectMask=b.dstSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;b.srcSubresource.layerCount=b.dstSubresource.layerCount=1;b.srcOffsets[1]={w,h,1};b.dstOffsets[0]={x,y,0};b.dstOffsets[1]={x+w,y+h,1};vkCmdBlitImage(vk.command,src.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,dst.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&b,VK_FILTER_NEAREST);if(vk.framebufferReadback){Transition(vk.command,dst.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);VkBufferImageCopy read{};read.bufferOffset=((VkDeviceSize)y*VRAM_WIDTH+x)*4;read.bufferRowLength=VRAM_WIDTH;read.bufferImageHeight=VRAM_HEIGHT;read.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;read.imageSubresource.layerCount=1;read.imageOffset={x,y,0};read.imageExtent={(uint32_t)w,(uint32_t)h,1};vkCmdCopyImageToBuffer(vk.command,dst.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,vk.framebufferReadback,1,&read);Transition(vk.command,dst.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);QueueFramebufferReadRect(x,y,w,h);}else Transition(vk.command,dst.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);Transition(vk.command,src.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);RememberFramebufferRect(x,y,w,h);}
        VkClearValue clear[2]{};VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};bi.renderPass=vk.loadRenderPass;bi.framebuffer=vk.framebuffers[vk.imageIndex];bi.renderArea.extent=vk.extent;bi.clearValueCount=2;bi.pClearValues=clear;vkCmdBeginRenderPass(vk.command,&bi,VK_SUBPASS_CONTENTS_INLINE);vk.inOffscreen=false;
    }
}

void GR_SetTexture(TextureID id,TexFormat f){vk.texFormat=f;vk.boundTexture=(g_dbg_texturelessMode?g_whiteTexture:id);if(vk.boundTexture>=kMaxTextures||!vk.textures[vk.boundTexture].alive)vk.boundTexture=g_whiteTexture;memset(vk.uniforms.framebufferRects,0,sizeof(vk.uniforms.framebufferRects));uint32_t dst=0;if(vk.displayFramebufferRectValid){const VkRect2D& r=vk.displayFramebufferRect;vk.uniforms.framebufferRects[dst][0]=(float)r.offset.x;vk.uniforms.framebufferRects[dst][1]=(float)r.offset.y;vk.uniforms.framebufferRects[dst][2]=(float)r.extent.width;vk.uniforms.framebufferRects[dst][3]=(float)r.extent.height;dst++;}for(uint32_t i=0;i<vk.framebufferRectCount&&dst<kFramebufferRegions;i++,dst++){const VkRect2D& r=vk.framebufferRects[i];vk.uniforms.framebufferRects[dst][0]=(float)r.offset.x;vk.uniforms.framebufferRects[dst][1]=(float)r.offset.y;vk.uniforms.framebufferRects[dst][2]=(float)r.extent.width;vk.uniforms.framebufferRects[dst][3]=(float)r.extent.height;}}
void GR_SetShader(ShaderID){} ShaderID GR_Shader_Compile(const char*){return 0;}
TextureID GR_CreateRGBATexture(int w,int h,u_char* data){if(vk.nextTexture>=kMaxTextures)return 0;TextureID id=vk.nextTexture++;std::vector<uint8_t> zero;if(!data){zero.resize((size_t)w*h*4);data=zero.data();}UploadTexture(vk.textures[id],data,(size_t)w*h*4,VK_FORMAT_R8G8B8A8_UNORM,w,h);UpdateDescriptor(id);return id;}
int GR_OverlayUploadRGBA(TextureID* texture,const unsigned char* rgba,int width,int height){
    if(!texture||!rgba||width<=0||height<=0||!vk.recording)return 0;if(!vk.overlayLogged){eprintf("*Vulkan developer overlay active\n");vk.overlayLogged=true;}SuspendMainPass();
    if(!*texture){if(vk.nextTexture>=kMaxTextures)return 0;*texture=vk.nextTexture++;}
    Texture& t=vk.textures[*texture];const VkDeviceSize bytes=(VkDeviceSize)width*height*4;
    if(!t.alive||t.width!=(uint32_t)width||t.height!=(uint32_t)height){ReleaseTextureForReplacement(t);if(!UploadTexture(t,rgba,(size_t)bytes,VK_FORMAT_R8G8B8A8_UNORM,width,height))return 0;t.linear=false;UpdateDescriptor(*texture);return 1;}
    return RecordTextureUpdate(t,rgba,bytes)?1:0;
}
void GR_OverlayDrawQuad(TextureID texture,float x0,float y0,float x1,float y1,float u0,float v0,float u1,float v1){
    if(!vk.recording||texture>=kMaxTextures||!vk.textures[texture].alive||!vk.overlayPipeline)return;ResumeMainPass();VkViewport viewport{0.0f,(float)vk.extent.height,(float)vk.extent.width,-(float)vk.extent.height,0.0f,1.0f};VkRect2D scissor{{0,0},vk.extent};vkCmdSetViewport(vk.command,0,1,&viewport);vkCmdSetScissor(vk.command,0,1,&scissor);vkCmdBindPipeline(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,vk.overlayPipeline);uint32_t offset=0;VkDescriptorSet descriptor=vk.textures[texture].descriptor;vkCmdBindDescriptorSets(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,vk.pipelineLayout,0,1,&descriptor,1,&offset);OverlayConstants push{{x0,y0,x1,y1},{u0,v0,u1,v1}};vkCmdPushConstants(vk.command,vk.pipelineLayout,VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(push),&push);vkCmdDraw(vk.command,6,1,0,0);
}
void GR_OverlayDrawLines(const float* xyRgb,int vertexCount){
    if(!vk.recording||!xyRgb||vertexCount<=0||!vk.linePipeline)return;VkDeviceSize bytes=(VkDeviceSize)vertexCount*5*sizeof(float);if(bytes>kOverlayLineBytes)return;ResumeMainPass();memcpy(vk.overlayLineMap,xyRgb,(size_t)bytes);VkViewport viewport{0.0f,(float)vk.extent.height,(float)vk.extent.width,-(float)vk.extent.height,0.0f,1.0f};VkRect2D scissor{{0,0},vk.extent};vkCmdSetViewport(vk.command,0,1,&viewport);vkCmdSetScissor(vk.command,0,1,&scissor);vkCmdBindPipeline(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,vk.linePipeline);VkDeviceSize offset=0;vkCmdBindVertexBuffers(vk.command,0,1,&vk.overlayLines,&offset);vkCmdDraw(vk.command,vertexCount,1,0,0);
}
int GR_UploadRGBATexture(TextureID* id,const u_char* data,int w,int h,int nearest,int){
    if(!id||!data||w<=0||h<=0)return 0;
    if(!*id){if(vk.nextTexture>=kMaxTextures)return 0;*id=vk.nextTexture++;}
    if(*id>=kMaxTextures)return 0;
    Texture& t=vk.textures[*id];ReleaseTextureForReplacement(t);t.linear=nearest==0;
    if(!UploadTexture(t,data,(size_t)w*h*4,VK_FORMAT_R8G8B8A8_UNORM,w,h))return 0;
    t.linear=nearest==0;UpdateDescriptor(*id);return 1;
}
void GR_DestroyTexture(TextureID id){if(id<kMaxTextures)ReleaseTextureForReplacement(vk.textures[id]);}
void GR_SetOverrideTextureSize(int w,int h,int ox,int oy,int hw,int hh){vk.uniforms.textureInfo[0]=1.f/w;vk.uniforms.textureInfo[1]=1.f/h;vk.uniforms.textureInfo[2]=(float)ox;vk.uniforms.textureInfo[3]=(float)oy;vk.uniforms.hiresInfo[0]=(hw>0)?std::min(.5f,.5f*w/hw):0;vk.uniforms.hiresInfo[1]=(hh>0)?std::min(.5f,.5f*h/hh):0;}
void GR_UpdateVertexBuffer(const GrVertex* v,int n){
    if(!v||n<=0||!vk.recording)return;if(vk.vertexSliceIndex>=vk.vertexSlices.size()){VertexSlice slice{};if(!CreateBuffer(kVertexBytes,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,slice.buffer,slice.memory,&slice.mapped))return;vk.vertexSlices.push_back(slice);eprintf("*Vulkan vertex pool expanded to %u batches\n",(unsigned)vk.vertexSlices.size());}VertexSlice& slice=vk.vertexSlices[vk.vertexSliceIndex++];slice.vertexCount=(uint32_t)std::min(n,MAX_VERTEX_BUFFER_SIZE);memcpy(slice.mapped,v,(size_t)slice.vertexCount*sizeof(GrVertex));vk.currentVertexSlice=&slice;VkDeviceSize offset=0;vkCmdBindVertexBuffers(vk.command,0,1,&slice.buffer,&offset);
}

static void CaptureRayTriangle(const GrVertex* vertex)
{
    if(vk.rtFrameMaterials.size()>=kMaxRayTriangles)return;
    for(int i=0;i<3;i++){
        if(vertex[i].ny<0.5f||vertex[i].nx>0.5f||
           !std::isfinite(vertex[i].vsx)||!std::isfinite(vertex[i].vsy)||!std::isfinite(vertex[i].vsz)||
           std::abs(vertex[i].vsx)>1000000.0f||std::abs(vertex[i].vsy)>1000000.0f||std::abs(vertex[i].vsz)>1000000.0f)return;
    }
    const float ax=vertex[1].vsx-vertex[0].vsx,ay=vertex[1].vsy-vertex[0].vsy,az=vertex[1].vsz-vertex[0].vsz;
    const float bx=vertex[2].vsx-vertex[0].vsx,by=vertex[2].vsy-vertex[0].vsy,bz=vertex[2].vsz-vertex[0].vsz;
    const float cx=ay*bz-az*by,cy=az*bx-ax*bz,cz=ax*by-ay*bx;
    if(cx*cx+cy*cy+cz*cz<1.0e-6f)return;

    for(int i=0;i<3;i++){
        vk.rtFramePositions.push_back(vertex[i].vsx);
        vk.rtFramePositions.push_back(vertex[i].vsy);
        vk.rtFramePositions.push_back(vertex[i].vsz);
    }

    RayMaterial material{};
    material.uv01[0]=(float)vertex[0].u+(float)vertex[0].tcx*0.5f;
    material.uv01[1]=(float)vertex[0].v+(float)vertex[0].tcy*0.5f;
    material.uv01[2]=(float)vertex[1].u+(float)vertex[1].tcx*0.5f;
    material.uv01[3]=(float)vertex[1].v+(float)vertex[1].tcy*0.5f;
    material.uv2Format[0]=(float)vertex[2].u+(float)vertex[2].tcx*0.5f;
    material.uv2Format[1]=(float)vertex[2].v+(float)vertex[2].tcy*0.5f;
    material.uv2Format[2]=(float)vk.texFormat;
    const int page=vertex[0].page,clut=vertex[0].clut;
    material.pageClut[0]=(float)(page&15)*64.0f+0.00025f;
    material.pageClut[1]=(float)(page/16)*256.0f+0.00025f;
    material.pageClut[2]=(float)(clut&63)/64.0f+0.00025f;
    material.pageClut[3]=(float)(clut/64)/512.0f+0.00025f;
    float* colors[3]={material.color0,material.color1,material.color2};
    for(int i=0;i<3;i++){
        const float bright=(float)vertex[i].bright;
        colors[i][0]=(float)vertex[i].r*(bright/255.0f);
        colors[i][1]=(float)vertex[i].g*(bright/255.0f);
        colors[i][2]=(float)vertex[i].b*(bright/255.0f);
        colors[i][3]=(float)vertex[i].a/255.0f;
    }
    vk.rtFrameMaterials.push_back(material);
}

void GR_DrawTriangles(int start,int tris){
    if(!vk.recording||!tris)return;if(g_cfg_rtgi&&vk.rayQueryAvailable&&vk.currentVertexSlice&&vk.boundTexture==g_vramTexture&&vk.blend==BM_NONE&&!vk.inOffscreen){const GrVertex* source=(const GrVertex*)vk.currentVertexSlice->mapped;uint32_t first=(uint32_t)std::max(start,0),end=std::min<uint32_t>(first+(uint32_t)tris*3,vk.currentVertexSlice->vertexCount);for(uint32_t i=first;i+2<end;i+=3)CaptureRayTriangle(source+i);}uint32_t slot=vk.uniformSlot++%kUniformSlots;vk.uniforms.fogColorStrength[0]=g_PsyX_FogColor[0];vk.uniforms.fogColorStrength[1]=g_PsyX_FogColor[1];vk.uniforms.fogColorStrength[2]=g_PsyX_FogColor[2];vk.uniforms.fogColorStrength[3]=g_PsyX_FogStrength;vk.uniforms.hiresInfo[2]=(g_cfg_psxDither&&!g_PsxDitherSuppressed)?1.f:0;vk.uniforms.hiresInfo[3]=(float)vk.texFormat;vk.uniforms.renderInfo[0]=(float)g_PsxUsePgxp;vk.uniforms.renderInfo[1]=g_PgxpFarWClamp;vk.uniforms.renderInfo[2]=(float)(g_PsxDitherSuppressed?(g_cfg_menuFilter?2:0):(g_cfg_bilinearFiltering?1:0));float size=g_PsyX_FlashlightFpsMode?g_PsyX_FlashlightSizeFps:g_PsyX_FlashlightSize;vk.uniforms.renderInfo[3]=1-size*(1-g_PsyX_FlashlightInnerCos);memcpy(vk.uniforms.lightPosRange,g_PsyX_FlashlightPos,12);vk.uniforms.lightPosRange[3]=g_PsyX_FlashlightRange;memcpy(vk.uniforms.lightDirStyle,g_PsyX_FlashlightDir,12);vk.uniforms.lightDirStyle[3]=0.0f;float intensity=g_PsyX_FlashlightFpsMode?g_PsyX_FlashlightIntensityFps:g_PsyX_FlashlightIntensity;for(int i=0;i<3;i++)vk.uniforms.lightColor[i]=g_PsyX_FlashlightColor[i]*intensity;vk.uniforms.lightColor[3]=1-size*(1-g_PsyX_FlashlightOuterCos);vk.uniforms.effectInfo[0]=(g_PsyX_UsePerPixelFlashlight&&g_PsyX_FlashlightActive)?1.0f:0.0f;vk.uniforms.effectInfo[1]=(float)g_PsyX_FlashlightStyle;vk.uniforms.effectInfo[2]=(float)g_PsxFogToBlack;vk.uniforms.effectInfo[3]=(g_PsyX_UseFlashlightShadows&&g_PsyX_UsePerPixelFlashlight&&g_PsyX_FlashlightActive&&g_PsyX_ShadowsAllowed&&!g_PsxPresentLastFrame&&vk.shadowView)?1.0f:0.0f;memcpy(vk.uniforms.shadowMatrix,vk.shadowLightMatrix,sizeof(vk.shadowLightMatrix));vk.uniforms.shadowParams[0]=vk.uniforms.effectInfo[3];vk.uniforms.shadowParams[1]=g_PsyX_FlashlightShadowBias;vk.uniforms.shadowParams[2]=g_PsyX_FlashlightShadowNormalOffset;vk.uniforms.shadowParams[3]=g_PsyX_FlashlightShadowStrength;vk.uniforms.shadowClip[0]=vk.shadowZNear;vk.uniforms.shadowClip[1]=vk.shadowZFar;vk.uniforms.shadowClip[2]=1.0f/(float)kShadowMapSize;vk.uniforms.shadowClip[3]=g_PsyX_FlashlightShadowFadeDist;memcpy((char*)vk.uniformMap+slot*vk.uniformStride,&vk.uniforms,sizeof(vk.uniforms));
    VkPipeline p=GetPipeline();vkCmdBindPipeline(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,p);vkCmdSetViewport(vk.command,0,1,&vk.viewport);VkExtent2D full=vk.inOffscreen?VkExtent2D{(uint32_t)vk.offscreenRect.w,(uint32_t)vk.offscreenRect.h}:vk.extent;VkRect2D s=vk.scissorEnabled?vk.scissor:VkRect2D{{0,0},full};
    const int maxW=(int)full.width,maxH=(int)full.height;
    const int sx=std::clamp(s.offset.x,0,maxW),sy=std::clamp(s.offset.y,0,maxH);
    const uint32_t sw=std::min(s.extent.width,(uint32_t)(maxW-sx)),sh=std::min(s.extent.height,(uint32_t)(maxH-sy));
    s.offset={sx,sy};s.extent={sw,sh};
    if(sw==0||sh==0)return;
    vkCmdSetScissor(vk.command,0,1,&s);float blend[4]={.25f,.25f,.25f,.25f};vkCmdSetBlendConstants(vk.command,blend);vkCmdSetDepthBias(vk.command,vk.polygonOffset,0.0f,0.0f);uint32_t offset=(uint32_t)(slot*vk.uniformStride);VkDescriptorSet set=vk.textures[vk.boundTexture].descriptor;const bool rtActive=GR_RayTracingEnabled()&&vk.rtCurrent.tlas&&!vk.inOffscreen&&vk.rtPipelineLayout;VkPipelineLayout layout=rtActive?vk.rtPipelineLayout:vk.pipelineLayout;vkCmdBindDescriptorSets(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,1,&set,1,&offset);if(rtActive)vkCmdBindDescriptorSets(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,1,1,&vk.rtDescriptor,0,nullptr);vkCmdDraw(vk.command,tris*3,1,start,0);
}

void GR_EnableDepth(int e){vk.depth=((e&&g_cfg_pgxpZBuffer)||g_PsyX_ForceItemDepth);}
void GR_SetDepthFuncAlways(int e){vk.depthAlways=e!=0;} void GR_SetStencilMode(int e){vk.stencil=e!=0;}
void GR_SetBlendMode(BlendMode b){vk.blend=b;g_PsxFogToBlack=(b==BM_ADD||b==BM_SUBTRACT||b==BM_ADD_QUATER_SOURCE);GR_EnableDepth(b==BM_NONE);}
void GR_SetPolygonOffset(float offset){vk.polygonOffset=offset;} void GR_SetWireframe(int e){vk.wire=e!=0&&vk.fillModeNonSolid;}
void GR_Clear(int,int,int,int,u_char r,u_char g,u_char b){
    if(!vk.recording)return;VkClearAttachment a[2]{};a[0].aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;a[1].aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT;a[1].clearValue.depthStencil={1,0};VkClearRect cr{};cr.rect.extent=vk.inOffscreen?VkExtent2D{(uint32_t)vk.offscreenRect.w,(uint32_t)vk.offscreenRect.h}:vk.extent;cr.layerCount=1;
    const bool pillarbox=(g_PcHorPlusEnabled&&g_PcWidescreenMode==0)||(!g_PcHorPlusEnabled&&g_PcMenuPillarbox);
    if(!vk.inOffscreen&&pillarbox&&g_windowWidth>0&&g_windowHeight>0&&(r|g|b)!=0&&(float)g_windowWidth/g_windowHeight>4.0f/3.0f){a[0].clearValue.color={{0,0,0,1}};vkCmdClearAttachments(vk.command,2,a,1,&cr);int contentWidth=(int)(g_windowHeight*(4.0f/3.0f)+0.5f);cr.rect.offset.x=(g_windowWidth-contentWidth)/2;cr.rect.extent.width=(uint32_t)contentWidth;}
    a[0].clearValue.color={{r/255.f,g/255.f,b/255.f,1}};vkCmdClearAttachments(vk.command,2,a,1,&cr);
}
void GR_ClearVRAM(int x,int y,int w,int h,u_char r,u_char g,u_char b){int x0=std::clamp(x,0,VRAM_WIDTH),y0=std::clamp(y,0,VRAM_HEIGHT),x1=std::clamp(x+w,0,VRAM_WIDTH),y1=std::clamp(y+h,0,VRAM_HEIGHT);for(int yy=y0;yy<y1;yy++)for(int xx=x0;xx<x1;xx++)vram[yy*VRAM_WIDTH+xx]=(r|(g<<5)|(b<<10));InvalidateFramebufferRects(x0,y0,x1-x0,y1-y0);GR_UpdateVRAM();}
void GR_CopyVRAM(unsigned short* src,int x,int y,int w,int h,int dx,int dy){if(w<=0||h<=0)return;int sourceStride=src?w:VRAM_WIDTH;if(!src){x=std::clamp(x,0,VRAM_WIDTH);y=std::clamp(y,0,VRAM_HEIGHT);w=std::min(w,VRAM_WIDTH-x);h=std::min(h,VRAM_HEIGHT-y);std::vector<unsigned short> tmp((size_t)w*h);for(int row=0;row<h;row++)memcpy(tmp.data()+row*w,vram+(y+row)*VRAM_WIDTH+x,w*2);GR_CopyVRAM(tmp.data(),0,0,w,h,dx,dy);return;}if(dx<0){int skip=-dx;x+=skip;w-=skip;dx=0;}if(dy<0){int skip=-dy;y+=skip;h-=skip;dy=0;}w=std::min(w,VRAM_WIDTH-dx);h=std::min(h,VRAM_HEIGHT-dy);if(w<=0||h<=0)return;const unsigned short* source=src+y*sourceStride+x;for(int row=0;row<h;row++)memcpy(vram+(dy+row)*VRAM_WIDTH+dx,source+row*sourceStride,w*2);InvalidateFramebufferRects(dx,dy,w,h);GR_UpdateVRAM();}
void GR_ReadVRAM(unsigned short* dst,int x,int y,int w,int h){for(int row=0;row<h;row++)memcpy(dst+row*w,vram+(y+row)*VRAM_WIDTH+x,w*2);}
void GR_UpdateVRAM(){if(vk.vramStagingMap)memcpy(vk.vramStagingMap,vram,sizeof(vram));}
void GR_DirectUploadVRAMRegion(int,int,int,int){GR_UpdateVRAM();}
void GR_ReadFramebufferDataToVRAM(){
    if(!vk.framebufferReadReadyCount||!vk.framebufferReadbackMap)return;
    if(g_PsxSkipFramebufferStore){vk.framebufferReadReadyCount=0;return;}
    const uint8_t* pixels=(const uint8_t*)vk.framebufferReadbackMap;const bool bgra=vk.colorFormat==VK_FORMAT_B8G8R8A8_UNORM||vk.colorFormat==VK_FORMAT_B8G8R8A8_SRGB;
    for(uint32_t region=0;region<vk.framebufferReadReadyCount;region++){
        const VkRect2D& rect=vk.framebufferReadReadyRects[region];const int x=rect.offset.x,y=rect.offset.y,w=(int)rect.extent.width,h=(int)rect.extent.height;
        for(int row=0;row<h;row++)for(int col=0;col<w;col++){
            const uint8_t* p=pixels+(((size_t)(y+row)*VRAM_WIDTH+x+col)*4);const uint16_t r=p[bgra?2:0]>>3,g=p[1]>>3,b=p[bgra?0:2]>>3;vram[(y+row)*VRAM_WIDTH+x+col]=(uint16_t)(r|(g<<5)|(b<<10)|((r|g|b)?0x8000:0));
        }
    }
    vk.framebufferReadReadyCount=0;
}
void GR_StoreFrameBuffer(int x,int y,int w,int h){if(g_PsxSkipFramebufferStore||w<=0||h<=0)return;x=std::clamp(x,0,VRAM_WIDTH);y=std::clamp(y,0,VRAM_HEIGHT);w=std::clamp(w,0,VRAM_WIDTH-x);h=std::clamp(h,0,VRAM_HEIGHT-y);vk.framebufferRect={{x,y},{(uint32_t)w,(uint32_t)h}};vk.framebufferStoreRequested=w>0&&h>0;}
void GR_SaveVRAM(const char* f,int x,int y,int w,int h,int readFramebuffer){if(!f||w<=0||h<=0)return;if(readFramebuffer)GR_ReadFramebufferDataToVRAM();x=std::clamp(x,0,VRAM_WIDTH);y=std::clamp(y,0,VRAM_HEIGHT);w=std::min(w,VRAM_WIDTH-x);h=std::min(h,VRAM_HEIGHT-y);uint8_t header[18]{};header[2]=2;header[12]=(uint8_t)w;header[13]=(uint8_t)(w>>8);header[14]=(uint8_t)h;header[15]=(uint8_t)(h>>8);header[16]=16;FILE* fp=fopen(f,"wb");if(!fp)return;fwrite(header,1,sizeof(header),fp);for(int row=h-1;row>=0;row--)fwrite(vram+(y+row)*VRAM_WIDTH+x,sizeof(uint16_t),w,fp);fclose(fp);} void GR_DumpVRAM(const char* f){GR_SaveVRAM(f,0,0,VRAM_WIDTH,VRAM_HEIGHT,1);}
static float ShadowDot(const float* a,const float* b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static void ShadowNormalize(float* v){float length=sqrtf(ShadowDot(v,v));if(length>1e-8f){v[0]/=length;v[1]/=length;v[2]/=length;}}
static void ShadowCross(const float* a,const float* b,float* r){r[0]=a[1]*b[2]-a[2]*b[1];r[1]=a[2]*b[0]-a[0]*b[2];r[2]=a[0]*b[1]-a[1]*b[0];}
static void ShadowLookAt(const float* eye,const float* center,const float* up,float* m){float f[3]={center[0]-eye[0],center[1]-eye[1],center[2]-eye[2]};ShadowNormalize(f);float s[3];ShadowCross(f,up,s);ShadowNormalize(s);float u[3];ShadowCross(s,f,u);m[0]=s[0];m[4]=s[1];m[8]=s[2];m[12]=-ShadowDot(s,eye);m[1]=u[0];m[5]=u[1];m[9]=u[2];m[13]=-ShadowDot(u,eye);m[2]=-f[0];m[6]=-f[1];m[10]=-f[2];m[14]=ShadowDot(f,eye);m[3]=m[7]=m[11]=0;m[15]=1;}
static void ShadowMultiply(const float* a,const float* b,float* result){for(int column=0;column<4;column++)for(int row=0;row<4;row++)result[column*4+row]=a[row]*b[column*4]+a[4+row]*b[column*4+1]+a[8+row]*b[column*4+2]+a[12+row]*b[column*4+3];}
static void BuildShadowMatrix(){float size=g_PsyX_FlashlightFpsMode?g_PsyX_FlashlightSizeFps:g_PsyX_FlashlightSize;float outer=std::clamp(1.0f-size*(1.0f-g_PsyX_FlashlightOuterCos),0.05f,0.999f);float fov=std::clamp(acosf(outer)*2.0f*1.25f,0.2f,2.9f);vk.shadowZNear=20.0f;vk.shadowZFar=std::max(vk.shadowZNear+1.0f,g_PsyX_FlashlightRange*1.3f);float projection[16]={};float f=1.0f/tanf(fov*0.5f);projection[0]=projection[5]=f;projection[10]=vk.shadowZFar/(vk.shadowZNear-vk.shadowZFar);projection[11]=-1.0f;projection[14]=(vk.shadowZFar*vk.shadowZNear)/(vk.shadowZNear-vk.shadowZFar);const float* source=g_PsyX_FlashlightFpsMode?g_PsyX_FlashlightShadowPos:g_PsyX_FlashlightPos;float eye[3]={source[0],source[1],source[2]},dir[3]={g_PsyX_FlashlightDir[0],g_PsyX_FlashlightDir[1],g_PsyX_FlashlightDir[2]};ShadowNormalize(dir);if(g_PsyX_FlashlightFpsMode&&g_PsyX_FlashlightShadowFpsDrop!=0){eye[0]-=dir[0]*g_PsyX_FlashlightShadowFpsDrop;eye[1]-=dir[1]*g_PsyX_FlashlightShadowFpsDrop;eye[2]-=dir[2]*g_PsyX_FlashlightShadowFpsDrop;}float center[3]={eye[0]+dir[0],eye[1]+dir[1],eye[2]+dir[2]},up[3]={0,1,0};if(fabsf(ShadowDot(dir,up))>0.99f){up[0]=1;up[1]=up[2]=0;}float view[16];ShadowLookAt(eye,center,up,view);ShadowMultiply(projection,view,vk.shadowLightMatrix);}
int GR_FlashlightShadowActive(){return g_PsyX_UseFlashlightShadows&&g_PsyX_UsePerPixelFlashlight&&g_PsyX_FlashlightActive&&g_PsyX_ShadowsAllowed&&!g_PsxPresentLastFrame&&vk.shadowPipeline;}
void GR_ShadowPassBegin(){if(!vk.recording||vk.inOffscreen||!vk.shadowPipeline)return;BuildShadowMatrix();vkCmdEndRenderPass(vk.command);VkClearValue clear{};clear.depthStencil={1.0f,0};VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};pass.renderPass=vk.shadowRenderPass;pass.framebuffer=vk.shadowFramebuffer;pass.renderArea.extent={kShadowMapSize,kShadowMapSize};pass.clearValueCount=1;pass.pClearValues=&clear;vkCmdBeginRenderPass(vk.command,&pass,VK_SUBPASS_CONTENTS_INLINE);VkViewport viewport{0,0,(float)kShadowMapSize,(float)kShadowMapSize,0,1};VkRect2D scissor{{0,0},{kShadowMapSize,kShadowMapSize}};vkCmdSetViewport(vk.command,0,1,&viewport);vkCmdSetScissor(vk.command,0,1,&scissor);vkCmdSetDepthBias(vk.command,1.0f,0.0f,1.5f);uint32_t slot=vk.uniformSlot++%kUniformSlots;memcpy(vk.uniforms.shadowMatrix,vk.shadowLightMatrix,sizeof(vk.shadowLightMatrix));memcpy((char*)vk.uniformMap+slot*vk.uniformStride,&vk.uniforms,sizeof(vk.uniforms));uint32_t offset=(uint32_t)(slot*vk.uniformStride);VkDescriptorSet descriptor=vk.textures[g_whiteTexture].descriptor;vkCmdBindPipeline(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,vk.shadowPipeline);vkCmdBindDescriptorSets(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,vk.pipelineLayout,0,1,&descriptor,1,&offset);vk.inShadow=true;}
void GR_ShadowPassDraw(int start,int count){if(vk.inShadow&&count>0)vkCmdDraw(vk.command,count,1,start,0);}
void GR_ShadowPassEnd(){if(!vk.inShadow)return;vkCmdEndRenderPass(vk.command);VkClearValue clear[2]{};VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};pass.renderPass=vk.loadRenderPass;pass.framebuffer=vk.framebuffers[vk.imageIndex];pass.renderArea.extent=vk.extent;pass.clearValueCount=2;pass.pClearValues=clear;vkCmdBeginRenderPass(vk.command,&pass,VK_SUBPASS_CONTENTS_INLINE);vk.inShadow=false;}
void GR_CaptureLastFrame(){if(!g_PsxPresentLastFrame){vk.captureRequested=true;FlushPreOverlayCopies();}}
void GR_PresentLastFrame(){
    if(!vk.recording||!vk.captureValid||!vk.captureTexture||!vk.textures[vk.captureTexture].alive||!vk.postPipeline||!vk.capturePostDescriptor)return;
    /* This is a framebuffer copy, not a PS1 primitive.  The PSX pipeline would
     * classify the quad as 3D and re-apply dither/quantization and cached draw
     * state, corrupting pause and no-map backdrops.  The fullscreen pipeline is
     * an exact nearest-sampled copy with depth, stencil and blending disabled. */
    ResumeMainPass();
    VkViewport viewport{0.0f,0.0f,(float)vk.extent.width,(float)vk.extent.height,0.0f,1.0f};VkRect2D scissor{{0,0},vk.extent};vkCmdSetViewport(vk.command,0,1,&viewport);vkCmdSetScissor(vk.command,0,1,&scissor);vkCmdBindPipeline(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,vk.postPipeline);vkCmdBindDescriptorSets(vk.command,VK_PIPELINE_BIND_POINT_GRAPHICS,vk.postPipelineLayout,0,1,&vk.capturePostDescriptor,0,nullptr);
    PostConstants constants{};constants.mode=0;constants.tonemapMode=0;constants.texelSize[0]=1.0f/std::max(1u,vk.extent.width);constants.texelSize[1]=1.0f/std::max(1u,vk.extent.height);vkCmdPushConstants(vk.command,vk.postPipelineLayout,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(constants),&constants);vkCmdDraw(vk.command,3,1,0,0);
}
void GR_InitPostProcess(){
    if(vk.postSetLayout)return;
    VkDescriptorSetLayoutBinding binding{0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};setInfo.bindingCount=1;setInfo.pBindings=&binding;if(!Check(vkCreateDescriptorSetLayout(vk.device,&setInfo,nullptr,&vk.postSetLayout),"post descriptor layout"))return;
    VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(PostConstants)};VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};layoutInfo.setLayoutCount=1;layoutInfo.pSetLayouts=&vk.postSetLayout;layoutInfo.pushConstantRangeCount=1;layoutInfo.pPushConstantRanges=&push;if(!Check(vkCreatePipelineLayout(vk.device,&layoutInfo,nullptr,&vk.postPipelineLayout),"post pipeline layout"))return;
    vk.postVert=Shader(g_post_vert_spv,sizeof(g_post_vert_spv));vk.postFrag=Shader(g_post_frag_spv,sizeof(g_post_frag_spv));EnsurePostTarget();CreatePostPipeline();
}
void GR_PostProcess(){
    if(!vk.recording||((g_cfg_postProcess<=0)&&(g_cfg_tonemap<=0)))return;
    if(!vk.postPipeline)GR_InitPostProcess();
    if(vk.postPipeline&&vk.postTexture.alive)vk.postRequested=true;
}
void GR_PushDebugLabel(const char* label){if(vk.recording&&vk.cmdBeginDebugLabel&&label){VkDebugUtilsLabelEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};info.pLabelName=label;vk.cmdBeginDebugLabel(vk.command,&info);}}
void GR_PopDebugLabel(){if(vk.recording&&vk.cmdEndDebugLabel)vk.cmdEndDebugLabel(vk.command);}
extern "C" void PsyX_ForceItemDepthBegin(){g_PsyX_ForceItemDepth=1;vk.depth=true;if(vk.recording){VkClearAttachment attachment{};attachment.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT;attachment.clearValue.depthStencil={1.0f,0};VkClearRect rect{};rect.rect.extent=vk.inOffscreen?VkExtent2D{(uint32_t)vk.offscreenRect.w,(uint32_t)vk.offscreenRect.h}:vk.extent;rect.layerCount=1;vkCmdClearAttachments(vk.command,1,&attachment,1,&rect);}}
extern "C" void PsyX_ForceItemDepthEnd(){g_PsyX_ForceItemDepth=0;vk.depth=false;}
int GR_ReadBackbuffer(void* output,int width,int height){
    if(!output||width!=(int)vk.extent.width||height!=(int)vk.extent.height||vk.recording)return 0;
    vkDeviceWaitIdle(vk.device);VkBuffer buffer{};VkDeviceMemory memory{};void* mapped=nullptr;size_t bytes=(size_t)width*height*4;
    if(!CreateBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,buffer,memory,&mapped))return 0;
    Immediate([&](VkCommandBuffer cmd){Transition(cmd,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);VkBufferImageCopy copy{};copy.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;copy.imageSubresource.layerCount=1;copy.imageExtent={(uint32_t)width,(uint32_t)height,1};vkCmdCopyImageToBuffer(cmd,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&copy);Transition(cmd,vk.images[vk.imageIndex],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);});
    uint8_t* src=(uint8_t*)mapped;uint8_t* dst=(uint8_t*)output;bool bgra=vk.colorFormat==VK_FORMAT_B8G8R8A8_UNORM||vk.colorFormat==VK_FORMAT_B8G8R8A8_SRGB;
    /* Match glReadPixels: row zero is the framebuffer's bottom row. Vulkan's
     * negative-height viewport makes image memory top-down, so reverse rows. */
    for(int y=0;y<height;y++){const uint8_t* sourceRow=src+(size_t)(height-1-y)*width*4;uint8_t* destRow=dst+(size_t)y*width*4;if(bgra){for(int x=0;x<width;x++){destRow[x*4]=sourceRow[x*4+2];destRow[x*4+1]=sourceRow[x*4+1];destRow[x*4+2]=sourceRow[x*4];destRow[x*4+3]=sourceRow[x*4+3];}}else memcpy(destRow,sourceRow,(size_t)width*4);}
    vkUnmapMemory(vk.device,memory);vkDestroyBuffer(vk.device,buffer,nullptr);vkFreeMemory(vk.device,memory,nullptr);return 1;
}
int GR_BlitRGB24Frame(const void* pixels,int width,int height){
    if(!pixels||width<=0||height<=0||!vk.device)return 0;
    const uint8_t* rgb=(const uint8_t*)pixels;
    vk.fmvRgba.resize((size_t)width*height*4);
    for(size_t i=0,n=(size_t)width*height;i<n;i++){
        vk.fmvRgba[i*4+0]=rgb[i*3+0];vk.fmvRgba[i*4+1]=rgb[i*3+1];
        vk.fmvRgba[i*4+2]=rgb[i*3+2];vk.fmvRgba[i*4+3]=255;
    }
    if(vk.recording)GR_SwapWindow();
    if(!vk.fmvTexture){if(vk.nextTexture>=kMaxTextures)return 0;vk.fmvTexture=vk.nextTexture++;}
    Texture& texture=vk.textures[vk.fmvTexture];
    const bool updateExisting=texture.alive&&texture.width==(uint32_t)width&&texture.height==(uint32_t)height;
    if(!updateExisting){
        ReleaseTextureForReplacement(texture);
        if(!UploadTexture(texture,vk.fmvRgba.data(),vk.fmvRgba.size(),VK_FORMAT_R8G8B8A8_UNORM,width,height))return 0;
        texture.linear=true;
        UpdateDescriptor(vk.fmvTexture);
    }

    GR_BeginScene();
    if(!vk.recording)return 0;
    if(updateExisting&&!RecordTextureUpdate(texture,vk.fmvRgba.data(),vk.fmvRgba.size()))return 0;
    ResumeMainPass();
    float videoAspect=(float)width/height,windowAspect=(float)g_windowWidth/g_windowHeight;
    int drawW=g_windowWidth,drawH=g_windowHeight;
    if(videoAspect>windowAspect)drawH=(int)(g_windowWidth/videoAspect+0.5f);
    else drawW=(int)(g_windowHeight*videoAspect+0.5f);
    short left=(short)((g_windowWidth-drawW)/2),top=(short)((g_windowHeight-drawH)/2);
    short right=(short)(left+drawW),bottom=(short)(top+drawH);
    GrVertex v[6]{};
    const short xy[6][2]={{left,top},{right,top},{left,bottom},{right,top},{right,bottom},{left,bottom}};
    const uint8_t uv[6][2]={{0,0},{255,0},{0,255},{255,0},{255,255},{0,255}};
    for(int i=0;i<6;i++){v[i].x=xy[i][0];v[i].y=xy[i][1];v[i].u=uv[i][0];v[i].v=uv[i][1];v[i].bright=1;v[i].r=v[i].g=v[i].b=v[i].a=255;}
    Ortho(0,(float)g_windowWidth,(float)g_windowHeight,0,-1,1);
    GR_UpdateVertexBuffer(v,6);GR_SetScissorState(0);GR_SetBlendMode(BM_NONE);GR_EnableDepth(0);GR_SetStencilMode(0);
    GR_SetTexture(vk.fmvTexture,(TexFormat)4);GR_SetOverrideTextureSize(1,1,0,0,0,0);GR_DrawTriangles(0,2);GR_SwapWindow();
    return 1;
}
void GR_ClearDepthStencil(){if(!vk.recording)return;VkClearAttachment a{};a.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT;a.clearValue.depthStencil={1,0};VkClearRect r{};r.rect.extent=vk.inOffscreen?VkExtent2D{(uint32_t)vk.offscreenRect.w,(uint32_t)vk.offscreenRect.h}:vk.extent;r.layerCount=1;vkCmdClearAttachments(vk.command,1,&a,1,&r);}
const char* GR_GetRendererName(){static char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(vk.physical,&p);strncpy(name,p.deviceName,sizeof(name)-1);return name;}
const char* GR_GetRendererVendor(){static char vendor[32];VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(vk.physical,&p);snprintf(vendor,sizeof(vendor),"Vulkan vendor 0x%04x",p.vendorID);return vendor;}
const char* GR_GetRendererVersion(){static char version[64];VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(vk.physical,&p);snprintf(version,sizeof(version),"Vulkan %u.%u.%u",VK_API_VERSION_MAJOR(p.apiVersion),VK_API_VERSION_MINOR(p.apiVersion),VK_API_VERSION_PATCH(p.apiVersion));return version;}

#endif
