/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "logical_device.hpp"

#include "command_execution_unit.hpp"
#include "logical_display.hpp"
#include "canvas_region_render_thread.hpp"
#include "vulkan_memory_object_uploader.hpp"

#include <math.h>
#include <unordered_set>

#include "_autogen/donut.vert.h"
#include "_autogen/donut.frag.h"

namespace vkdd {
struct DeallocationContainer
{
  FrameIndex                   m_frameIndex;
  VulkanMemoryPool::Allocation m_rawAllocation;
  BufferAllocation             m_bufferAllocation;
  ImageAllocation              m_imageAllocation;

  bool operator<(DeallocationContainer const& other) const { return m_frameIndex < other.m_frameIndex; }
};

LogicalDevice::LogicalDevice(vk::Instance instance, uint32_t devGroupIdx)
    : m_instance(instance)
    , m_devGroupIdx(devGroupIdx)
    , m_frameIndex(0)
{
  std::vector<vk::PhysicalDeviceGroupProperties> devGroups = m_instance.enumeratePhysicalDeviceGroups();
  assert(m_devGroupIdx < devGroups.size());
  vk::PhysicalDeviceGroupProperties devGroup = devGroups[m_devGroupIdx];
  for(uint32_t i = 0; i < devGroup.physicalDeviceCount; ++i)
  {
    m_physicalDevices.emplace_back(devGroup.physicalDevices[i]);
  }
  m_perSubDeviceMemPools = std::vector<MemPoolCollection>(m_physicalDevices.size());
}

LogicalDevice::~LogicalDevice() {}

LogicalDisplay* LogicalDevice::enableDisplay(class Scene const& scene, vk::DisplayKHR display, CanvasRegion displayRegionOnCanvas)
{
  for(UniqueLogicalDisplay const& logicalDisplay : m_logicalDisplays)
  {
    if(logicalDisplay->getDisplay() == display)
    {
      LOGE("Tried to enable display twice.\n");
      return nullptr;
    }
  }
  // vk_ddisplay
  // first find all physical device indices which report the desired display
  std::vector<DeviceIndex> deviceIndices;
  for(DeviceIndex devIdx = 0; devIdx < m_physicalDevices.size(); ++devIdx)
  {
    vk::PhysicalDevice                    dev      = m_physicalDevices[devIdx];
    std::vector<vk::DisplayPropertiesKHR> displays = dev.getDisplayPropertiesKHR();
    for(vk::DisplayPropertiesKHR dispProps : displays)
    {
      if(dispProps.display == display)
      {
        deviceIndices.emplace_back(devIdx);
        break;
      }
    }
  }
  if(deviceIndices.empty())
  {
    LOGE("Given display is not connected to any physical device of this device group.");
    return nullptr;
  }
  // vk_ddisplay
  // create the logical display and provide the sub device indices
  UniqueLogicalDisplay logicalDisplay = std::make_unique<LogicalDisplay>(*this, display, displayRegionOnCanvas);
  if(!logicalDisplay->init(scene, deviceIndices))
  {
    LOGE("Initialization of logical display failed.\n");
    return nullptr;
  }
  return m_logicalDisplays.emplace_back(std::move(logicalDisplay)).get();
}

void LogicalDevice::createDonutPipeline()
{
  m_donutVert = m_device->createShaderModuleUnique({{}, sizeof(donut_vert), donut_vert});
  m_donutFrag = m_device->createShaderModuleUnique({{}, sizeof(donut_frag), donut_frag});
  vk::PipelineCacheCreateInfo pipelineCacheCreateInfo({});
  m_donutPipelineCache = m_device->createPipelineCacheUnique(pipelineCacheCreateInfo);
  vk::PipelineShaderStageCreateInfo donutVertStage({}, vk::ShaderStageFlagBits::eVertex, m_donutVert.get(), "main");
  vk::PipelineShaderStageCreateInfo donutFragStage({}, vk::ShaderStageFlagBits::eFragment, m_donutFrag.get(), "main");
  std::vector<vk::PipelineShaderStageCreateInfo> stages = {donutVertStage, donutFragStage};
  vk::VertexInputBindingDescription perVertexBindingDesc(0, sizeof(DefaultVertex), vk::VertexInputRate::eVertex);
  vk::VertexInputBindingDescription perInstanceBindingDesc(1, sizeof(DefaultInstance), vk::VertexInputRate::eInstance);
  std::vector<vk::VertexInputBindingDescription> vertexInputBindingDescs = {perVertexBindingDesc, perInstanceBindingDesc};
  vk::VertexInputAttributeDescription vertexPosDesc(0, 0, vk::Format::eR32G32B32Sfloat, 0 * sizeof(float));
  vk::VertexInputAttributeDescription vertexNormalDesc(1, 0, vk::Format::eR32G32B32Sfloat, 3 * sizeof(float));
  vk::VertexInputAttributeDescription vertexTexDesc(2, 0, vk::Format::eR32G32Sfloat, 6 * sizeof(float));
  vk::VertexInputAttributeDescription instanceModel0Desc(3, 1, vk::Format::eR32G32B32A32Sfloat, 0 * sizeof(float));
  vk::VertexInputAttributeDescription instanceModel1Desc(4, 1, vk::Format::eR32G32B32A32Sfloat, 4 * sizeof(float));
  vk::VertexInputAttributeDescription instanceModel2Desc(5, 1, vk::Format::eR32G32B32A32Sfloat, 8 * sizeof(float));
  vk::VertexInputAttributeDescription instanceModel3Desc(6, 1, vk::Format::eR32G32B32A32Sfloat, 12 * sizeof(float));
  vk::VertexInputAttributeDescription instanceInvModel0Desc(7, 1, vk::Format::eR32G32B32A32Sfloat, 16 * sizeof(float));
  vk::VertexInputAttributeDescription instanceInvModel1Desc(8, 1, vk::Format::eR32G32B32A32Sfloat, 20 * sizeof(float));
  vk::VertexInputAttributeDescription instanceInvModel2Desc(9, 1, vk::Format::eR32G32B32A32Sfloat, 24 * sizeof(float));
  vk::VertexInputAttributeDescription instanceInvModel3Desc(10, 1, vk::Format::eR32G32B32A32Sfloat, 28 * sizeof(float));
  vk::VertexInputAttributeDescription instanceColorGradientIdxDesc(11, 1, vk::Format::eR32Uint, 32 * sizeof(float));
  vk::VertexInputAttributeDescription instanceShellHeightDesc(12, 1, vk::Format::eR32Sfloat, 33 * sizeof(float));
  vk::VertexInputAttributeDescription instanceExtrusionDesc(13, 1, vk::Format::eR32Sfloat, 34 * sizeof(float));
  std::vector<vk::VertexInputAttributeDescription> vertexInputAttributeDescs = {
      vertexPosDesc,           vertexNormalDesc,      vertexTexDesc,         instanceModel0Desc,
      instanceModel1Desc,      instanceModel2Desc,    instanceModel3Desc,    instanceInvModel0Desc,
      instanceInvModel1Desc,   instanceInvModel2Desc, instanceInvModel3Desc, instanceColorGradientIdxDesc,
      instanceShellHeightDesc, instanceExtrusionDesc};
  vk::PipelineVertexInputStateCreateInfo   vertexInputState({}, vertexInputBindingDescs, vertexInputAttributeDescs);
  vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState({}, vk::PrimitiveTopology::eTriangleStrip, true);
  vk::PipelineViewportStateCreateInfo      viewportState({}, 1, nullptr, 1, nullptr);
  vk::PipelineRasterizationStateCreateInfo rasterizationState({}, false, false, vk::PolygonMode::eFill,
                                                              vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise);
  rasterizationState.setLineWidth(1.0f);
  vk::PipelineMultisampleStateCreateInfo  multisampleState({}, vk::SampleCountFlagBits::e1);
  vk::PipelineDepthStencilStateCreateInfo depthStencilState({}, true, true, vk::CompareOp::eLess, false, false);
  vk::PipelineColorBlendAttachmentState   attachment;
  attachment.setColorWriteMask(vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags);
  vk::PipelineColorBlendStateCreateInfo colorBlendState({}, false, vk::LogicOp::eClear, attachment);
  std::vector<vk::DynamicState>         dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo    dynamicState({}, dynamicStates);
  vk::PushConstantRange                 pushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(GlobalData));
  vk::PipelineLayoutCreateInfo          pipelineLayoutCreateInfo({}, {}, pushConstantRange);
  m_donutPipelineLayout = m_device->createPipelineLayoutUnique(pipelineLayoutCreateInfo);
  vk::GraphicsPipelineCreateInfo donutPipelineCreateInfo({}, stages, &vertexInputState, &inputAssemblyState, {},
                                                         &viewportState, &rasterizationState, &multisampleState,
                                                         &depthStencilState, &colorBlendState, &dynamicState,
                                                         m_donutPipelineLayout.get(), m_donutRenderPass.get(), 0);
  auto donutPipelineResult = m_device->createGraphicsPipelineUnique(m_donutPipelineCache.get(), donutPipelineCreateInfo);
  assert(donutPipelineResult.result == vk::Result::eSuccess);
  m_donutPipeline = std::move(donutPipelineResult.value);
}

TriangleMesh* LogicalDevice::getDonutTriangleMesh(DeviceIndex deviceIndex, uint32_t baseNumTesselations)
{
  std::lock_guard guard(m_donutTriMeshesMtx);
  auto            findIt = m_donutTriMeshes[deviceIndex].find(baseNumTesselations);
  if(findIt != m_donutTriMeshes[deviceIndex].end())
  {
    return findIt->second.get();
  }
  TriangleMesh* triMesh = m_donutTriMeshes[deviceIndex]
                              .emplace(baseNumTesselations, std::make_unique<TriangleMesh>(*this, deviceIndex))
                              .first->second.get();
  triMesh->buildTorus(baseNumTesselations, 2 * baseNumTesselations);
  return triMesh;
}

MemTypeIndex LogicalDevice::getMemoryTypeIndex(DeviceIndex deviceIndex, uint32_t memoryTypeBits, vk::MemoryPropertyFlags memPropFlags)
{
  vk::PhysicalDeviceMemoryProperties memProps = m_physicalDevices[deviceIndex].getMemoryProperties();
  for(uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
  {
    if((memoryTypeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & memPropFlags) == memPropFlags)
    {
      return i;
    }
  }
  LOGE("Failed to allocate device memory.\n");
  return (uint32_t)-1;
}

VulkanMemoryPool::Allocation LogicalDevice::allocateHostVisibleDeviceMemory(vk::MemoryRequirements memReqs,
                                                                            void const*            initialData,
                                                                            size_t                 initialDataSize)
{
  vk::MemoryPropertyFlags memPropFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
  VulkanMemoryPool::Allocation allocation = this->allocateDeviceMemory({}, memReqs, memPropFlags);
  if(initialData != nullptr && initialDataSize != 0)
  {
    void* mapped = m_device->mapMemory(allocation.devMem(), allocation.devMemOffset(), initialDataSize);
    memcpy(mapped, initialData, initialDataSize);
    m_device->unmapMemory(allocation.devMem());
  }
  return std::move(allocation);
}

VulkanMemoryPool::Allocation LogicalDevice::allocateStagingMemory(size_t size, size_t alignment)
{
  return m_stagingMemPool->alloc(size, alignment);
}

VulkanMemoryPool::Allocation LogicalDevice::allocateDeviceMemory(OptionalDeviceIndex     deviceIndex,
                                                                 vk::MemoryRequirements  memReqs,
                                                                 vk::MemoryPropertyFlags memPropFlags)
{
  MemTypeIndex memTypeIdx = this->getMemoryTypeIndex(deviceIndex.value_or(0), memReqs.memoryTypeBits, memPropFlags);
  return this->getMemPool(deviceIndex, memTypeIdx)->alloc(memReqs.size, memReqs.alignment);
}

BufferAllocation LogicalDevice::allocateBuffer(OptionalDeviceIndex deviceIndex, vk::BufferCreateInfo createInfo, vk::MemoryPropertyFlags memPropFlags)
{
  vk::UniqueBuffer        buffer  = m_device->createBufferUnique(createInfo);
  vk::MemoryRequirements2 memReqs = m_device->getBufferMemoryRequirements(buffer.get());
  VulkanMemoryPool::Allocation allocation = this->allocateDeviceMemory(deviceIndex, memReqs.memoryRequirements, memPropFlags);
  m_device->bindBufferMemory(buffer.get(), allocation.devMem(), allocation.devMemOffset());
  return {std::move(buffer), std::move(allocation)};
}

ImageAllocation LogicalDevice::allocateImage(OptionalDeviceIndex deviceIndex, vk::ImageCreateInfo createInfo, vk::MemoryPropertyFlags memPropFlags)
{
  vk::UniqueImage         image   = m_device->createImageUnique(createInfo);
  vk::MemoryRequirements2 memReqs = m_device->getImageMemoryRequirements(image.get());
  VulkanMemoryPool::Allocation allocation = this->allocateDeviceMemory(deviceIndex, memReqs.memoryRequirements, memPropFlags);
  m_device->bindImageMemory(image.get(), allocation.devMem(), allocation.devMemOffset());
  return {std::move(image), std::move(allocation)};
}

BufferAllocation LogicalDevice::allocateStagingBuffer(vk::BufferCreateInfo createInfo)
{
  vk::UniqueBuffer             buffer     = m_device->createBufferUnique(createInfo);
  vk::MemoryRequirements       memReqs    = m_device->getBufferMemoryRequirements(buffer.get());
  VulkanMemoryPool::Allocation allocation = this->allocateStagingMemory(memReqs.size, memReqs.alignment);
  m_device->bindBufferMemory(buffer.get(), allocation.devMem(), allocation.devMemOffset());
  return {std::move(buffer), std::move(allocation)};
}

void LogicalDevice::scheduleForDeallocation(VulkanMemoryPool::Allocation allocation, uint32_t numFramesToKeepAlive)
{
  this->scheduleForDeallocation({m_frameIndex + numFramesToKeepAlive, std::move(allocation)});
}

void LogicalDevice::scheduleForDeallocation(BufferAllocation allocation, uint32_t numFramesToKeepAlive)
{
  this->scheduleForDeallocation({m_frameIndex + numFramesToKeepAlive, {}, std::move(allocation)});
}

void LogicalDevice::scheduleForDeallocation(ImageAllocation allocation, uint32_t numFramesToKeepAlive)
{
  this->scheduleForDeallocation({m_frameIndex + numFramesToKeepAlive, {}, {}, std::move(allocation)});
}

void LogicalDevice::scheduleForDeallocation(DeallocationContainer deallocation)
{
  std::lock_guard guard(m_deallocationQueueMtx);
  auto            lbIt = std::lower_bound(m_deallocationQueue.begin(), m_deallocationQueue.end(), deallocation);
  m_deallocationQueue.insert(lbIt, std::move(deallocation));
}

std::optional<uint32_t> LogicalDevice::getQueueFamilyIndex(vk::QueueFlags flags, std::unordered_set<uint32_t> excludeQueueFamilyIndices)
{
  std::unordered_set<uint32_t> candidates;
  for(vk::PhysicalDevice physicalDevice : m_physicalDevices)
  {
    std::vector<vk::QueueFamilyProperties> queueFamilyProps = physicalDevice.getQueueFamilyProperties();
    for(uint32_t i = 0; i < queueFamilyProps.size(); ++i)
    {
      if(excludeQueueFamilyIndices.find(i) == excludeQueueFamilyIndices.end())
      {
        bool desiredFlagsOnAllDevices = true;
        for(vk::PhysicalDevice otherPhysicalDevice : m_physicalDevices)
        {
          std::vector<vk::QueueFamilyProperties> otherQueueFamilyProps = physicalDevice.getQueueFamilyProperties();
          desiredFlagsOnAllDevices &= i < otherQueueFamilyProps.size() && (otherQueueFamilyProps[i].queueFlags & flags);
        }
        if(desiredFlagsOnAllDevices)
        {
          candidates.insert(i);
        }
      }
    }
  }
  if(candidates.empty())
  {
    LOGE("No common queue family index found.\n");
    return {};
  }
  return *candidates.begin();
}

vk::Queue LogicalDevice::getQueue(uint32_t queueFamilyIndex) const
{
  auto findIt = m_queues.find(queueFamilyIndex);
  return findIt == m_queues.end() ? nullptr : findIt->second;
}

bool LogicalDevice::start()
{
  // starting a logical device will create all its resources, including its vkDevice, queues, memory pools, and render
  // pass. at last the logical displays and their render contexts are started as well
  std::optional<uint32_t> graphicsQueueFamilyIndex = this->getQueueFamilyIndex(vk::QueueFlagBits::eGraphics, {});
  if(!graphicsQueueFamilyIndex.has_value())
  {
    LOGE("No graphics queue family index.\n");
    return false;
  }
  std::optional<uint32_t> transferQueueFamilyIndex =
      this->getQueueFamilyIndex(vk::QueueFlagBits::eTransfer, {graphicsQueueFamilyIndex.value()});
  if(!transferQueueFamilyIndex.has_value())
  {
    LOGE("No dedicated transfer queue family index.\n");
    return false;
  }
  std::optional<uint32_t> framebufferTransferQueueFamilyIndex =
      this->getQueueFamilyIndex(vk::QueueFlagBits::eTransfer,
                                {graphicsQueueFamilyIndex.value(), transferQueueFamilyIndex.value()});
  if(!framebufferTransferQueueFamilyIndex.has_value())
  {
    LOGE("No two dedicated transfer queue family indices.\n");
    return false;
  }
  m_graphicsQueueFamilyIndex            = graphicsQueueFamilyIndex.value();
  m_transferQueueFamilyIndex            = transferQueueFamilyIndex.value();
  m_framebufferTransferQueueFamilyIndex = framebufferTransferQueueFamilyIndex.value();
  LOGI("Queue family indices - graphics: %d, transfer: %d, fb transfer: %d.\n", m_graphicsQueueFamilyIndex,
       m_transferQueueFamilyIndex, m_framebufferTransferQueueFamilyIndex);

  std::vector<float>                     queuePriorities     = {1.0f};
  std::vector<vk::DeviceQueueCreateInfo> devQueueCreateInfos = {
      {{}, m_graphicsQueueFamilyIndex, queuePriorities},
      {{}, m_transferQueueFamilyIndex, queuePriorities},
      {{}, m_framebufferTransferQueueFamilyIndex, queuePriorities},
  };
  std::vector<char const*>                    enabledExtensions = {"VK_KHR_swapchain", "VK_NV_acquire_winrt_display"};
  vk::PhysicalDeviceSynchronization2Features  synchronization2Features(true);
  vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures(true, &synchronization2Features);
  vk::DeviceGroupDeviceCreateInfo             devGroupDevCreateInfo(m_physicalDevices, &timelineSemaphoreFeatures);
  vk::DeviceCreateInfo devCreateInfo({}, devQueueCreateInfos, {}, enabledExtensions, nullptr, &devGroupDevCreateInfo);
  m_device                                        = m_physicalDevices.front().createDeviceUnique(devCreateInfo);
  m_queues[m_graphicsQueueFamilyIndex]            = m_device->getQueue(m_graphicsQueueFamilyIndex, 0);
  m_queues[m_transferQueueFamilyIndex]            = m_device->getQueue(m_transferQueueFamilyIndex, 0);
  m_queues[m_framebufferTransferQueueFamilyIndex] = m_device->getQueue(m_framebufferTransferQueueFamilyIndex, 0);
  for(UniqueCommandExecutionUnit& cmdExecUnit : m_cmdExecUnits)
  {
    cmdExecUnit = std::make_unique<CommandExecutionUnit>(*this);
  }
  m_uploader = std::make_unique<VulkanMemoryObjectUploader>(*this);

  MemTypeIndex stagingMemTypeIdx =
      this->getMemoryTypeIndex(0, ~0, vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible);
  m_stagingMemPool = std::make_unique<VulkanMemoryPool>(m_device.get(), DeviceMask(), stagingMemTypeIdx, true);

  std::vector<vk::SurfaceFormatKHR> commonSurfaceFormats;
  if(!m_logicalDisplays.empty())
  {
    m_logicalDisplays.front()->querySurfaceFormats(commonSurfaceFormats);
  }
  for(auto const& logicalDisplay : m_logicalDisplays)
  {
    std::vector<vk::SurfaceFormatKHR> surfaceFormats;
    logicalDisplay->querySurfaceFormats(surfaceFormats);
    commonSurfaceFormats.erase(std::remove_if(commonSurfaceFormats.begin(), commonSurfaceFormats.end(),
                                              [&](auto const& fmt) {
                                                return std::find(surfaceFormats.begin(), surfaceFormats.end(), fmt)
                                                       == surfaceFormats.end();
                                              }),
                               commonSurfaceFormats.end());
  }
  if(commonSurfaceFormats.empty())
  {
    LOGE("No common surface format for shared display.\n");
    return false;
  }
  vk::Format           preferredFormat     = vk::Format::eB8G8R8A8Unorm;
  vk::ColorSpaceKHR    preferredColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
  vk::SurfaceFormatKHR surfFormat          = commonSurfaceFormats.front();
  for(vk::SurfaceFormatKHR fmt : commonSurfaceFormats)
  {
    if(fmt.format == preferredFormat && fmt.colorSpace == preferredColorSpace)
    {
      surfFormat = fmt;
      break;
    }
  }
  vk::AttachmentDescription colorAttachment({}, surfFormat.format, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear);
  colorAttachment.setInitialLayout(vk::ImageLayout::eColorAttachmentOptimal);
  colorAttachment.setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);
  vk::AttachmentDescription depthStencilAttachment({}, vk::Format::eD24UnormS8Uint, vk::SampleCountFlagBits::e1,
                                                   vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
                                                   vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore);
  depthStencilAttachment.setInitialLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);
  depthStencilAttachment.setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);
  std::vector<vk::AttachmentDescription> attachments = {colorAttachment, depthStencilAttachment};
  vk::AttachmentReference                colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);
  vk::AttachmentReference                depthStencilAttachmentRef(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
  vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, colorAttachmentRef, {}, &depthStencilAttachmentRef);
  vk::RenderPassCreateInfo renderPassCreateInfo({}, attachments, subpass);
  m_donutRenderPass = m_device->createRenderPassUnique(renderPassCreateInfo);

  this->createDonutPipeline();

  for(uint32_t i = 0; i < m_logicalDisplays.size(); ++i)
  {
    if(!m_logicalDisplays[i]->start(surfFormat, m_donutRenderPass.get()))
    {
      LOGI("Starting of rendering on display %d failed.\n", i);
    }
  }
  return true;
}

VulkanMemoryPool* LogicalDevice::getMemPool(OptionalDeviceIndex deviceIndex, MemTypeIndex memTypeIdx)
{
  std::lock_guard guard(m_memPoolsMtx);
  MemPoolCollection& memPoolCollection = deviceIndex.has_value() ? m_perSubDeviceMemPools[deviceIndex.value()] : m_globalMemPools;
  if(auto findIt = memPoolCollection.find(memTypeIdx); findIt != memPoolCollection.end())
  {
    return findIt->second.get();
  }
  DeviceMask deviceMask = deviceIndex.has_value() ? DeviceMask::ofSingleDevice(deviceIndex.value()) : DeviceMask();
  return memPoolCollection
      .emplace(memTypeIdx, std::make_unique<VulkanMemoryPool>(m_device.get(), deviceMask, memTypeIdx, false))
      .first->second.get();
}

void LogicalDevice::render()
{
  CommandExecutionUnit& cmdExecUnit = *m_cmdExecUnits[m_frameIndex % NUM_QUEUED_FRAMES];
  cmdExecUnit.waitForIdleAndReset();

  m_uploader->prepare(cmdExecUnit);
  for(auto const& logicalDisplay : m_logicalDisplays)
  {
    logicalDisplay->renderFrameAsync(cmdExecUnit);
  }
  std::vector<vk::Semaphore>    waitSems;
  std::vector<vk::SwapchainKHR> swapchains;
  std::vector<uint32_t>         imageIndices;
  for(auto const& logicalDisplay : m_logicalDisplays)
  {
    std::optional<LogicalDisplay::PresentData> presentData = logicalDisplay->finishFrameRendering(cmdExecUnit);
    if(presentData.has_value())
    {
      waitSems.emplace_back(presentData.value().m_waitSem);
      swapchains.emplace_back(presentData.value().m_swapchain);
      imageIndices.emplace_back(presentData.value().m_imageIndex);
    }
  }
  m_uploader->finish();
  cmdExecUnit.submit();

  // vk_ddisplay
  // all displays of this logical device can be presented at once
  if(!swapchains.empty())
  {
    vk::PresentInfoKHR presentInfo(waitSems, swapchains, imageIndices);
    vk::Result         presentResult = this->getQueue(m_graphicsQueueFamilyIndex).presentKHR(presentInfo);
    if(presentResult != vk::Result::eSuccess)
    {
      LOGW("presentKHR() failed.\n");
    }
  }

  auto it = m_deallocationQueue.begin();
  while(it != m_deallocationQueue.end() && it->m_frameIndex < m_frameIndex)
  {
    ++it;
  }
  m_deallocationQueue.erase(m_deallocationQueue.begin(), it);
  ++m_frameIndex;
}

void LogicalDevice::interrupt()
{
  m_cmdExecUnits[(m_frameIndex + NUM_QUEUED_FRAMES - 1) % NUM_QUEUED_FRAMES]->waitForIdle();
  for(auto const& logicalDisplay : m_logicalDisplays)
  {
    logicalDisplay->interrupt();
  }
  m_deallocationQueue.clear();
}

void LogicalDevice::join()
{
  for(auto const& logicalDisplay : m_logicalDisplays)
  {
    logicalDisplay->join();
  }
}
}  // namespace vkdd
