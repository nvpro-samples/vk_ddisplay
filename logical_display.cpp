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

#include "logical_display.hpp"

#include "canvas_region_render_thread.hpp"
#include "command_execution_unit.hpp"
#include "logical_device.hpp"

namespace vkdd {
bool contains(vk::Rect2D const& rect, int32_t x, int32_t y)
{
  return rect.offset.x <= x && x < (int32_t)(rect.offset.x + rect.extent.width) && rect.offset.y <= y
         && y < (int32_t)(rect.offset.y + rect.extent.height);
}

LogicalDisplay::LogicalDisplay(LogicalDevice& logicalDevice, vk::DisplayKHR display, CanvasRegion displayRegionOnCanvas)
    : m_display(display)
    , m_displayRegionOnCanvas(displayRegionOnCanvas)
    , m_logicalDevice(logicalDevice)
{
}

LogicalDisplay::~LogicalDisplay() {}

bool LogicalDisplay::init(class Scene const& scene, std::vector<DeviceIndex> const& deviceIndices)
{
  if(deviceIndices.empty())
  {
    LOGE("No physical device indices given.\n");
    return false;
  }
  vk::PhysicalDevice initialPhysicalDevice = m_logicalDevice.getPhysicalDevice(deviceIndices.front());
  initialPhysicalDevice.acquireWinrtDisplayNV(m_display);

  // select display mode with largest visible region and refresh rate
  std::vector<vk::DisplayModePropertiesKHR> allDisplayModeProps = initialPhysicalDevice.getDisplayModePropertiesKHR(m_display);
  vk::DisplayModePropertiesKHR displayModeProps =
      *std::max_element(allDisplayModeProps.begin(), allDisplayModeProps.end(),
                        [](vk::DisplayModePropertiesKHR a, vk::DisplayModePropertiesKHR b) {
                          return a.parameters.visibleRegion.width != b.parameters.visibleRegion.width ?
                                     a.parameters.visibleRegion.width < b.parameters.visibleRegion.width :
                                 a.parameters.visibleRegion.height != b.parameters.visibleRegion.height ?
                                     a.parameters.visibleRegion.height < b.parameters.visibleRegion.height :
                                     a.parameters.refreshRate < b.parameters.refreshRate;
                        });
  m_surfaceSize = displayModeProps.parameters.visibleRegion;
  std::vector<vk::DisplayPlanePropertiesKHR> deviceDisplayPlaneProps = initialPhysicalDevice.getDisplayPlanePropertiesKHR();
  std::optional<uint32_t> foundPlaneIdx;
  for(uint32_t planeIdx = 0; planeIdx < deviceDisplayPlaneProps.size(); ++planeIdx)
  {
    if(deviceDisplayPlaneProps[planeIdx].currentDisplay == m_display)
    {
      foundPlaneIdx = planeIdx;
      break;
    }
  }
  assert(foundPlaneIdx.has_value());

  uint32_t                        planeIdx = foundPlaneIdx.value();
  uint32_t                        stackIdx = deviceDisplayPlaneProps[planeIdx].currentStackIndex;
  vk::DisplayPlaneCapabilitiesKHR planeCaps =
      initialPhysicalDevice.getDisplayPlaneCapabilitiesKHR(displayModeProps.displayMode, planeIdx);
  if(!(planeCaps.supportedAlpha & vk::DisplayPlaneAlphaFlagBitsKHR::eOpaque))
  {
    LOGE("Display plane does not support opaque alpha.\n");
    return false;
  }
  vk::DisplaySurfaceCreateInfoKHR displaySurfaceCreateInfo({}, displayModeProps.displayMode, planeIdx, stackIdx,
                                                           vk::SurfaceTransformFlagBitsKHR::eIdentity, 1.0f,
                                                           vk::DisplayPlaneAlphaFlagBitsKHR::eOpaque,
                                                           displayModeProps.parameters.visibleRegion);
  m_surface = m_logicalDevice.vkInstance().createDisplayPlaneSurfaceKHRUnique(displaySurfaceCreateInfo);

  // vk_ddisplay
  // there is one dedicated render context for each device rendering to the desired display
  for(DeviceIndex devIdx : deviceIndices)
  {
    this->pushRenderContext(scene, devIdx);
  }
  return true;
}

void LogicalDisplay::pushRenderContext(class Scene const& scene, DeviceIndex deviceIndex)
{
  // vk_ddisplay
  // each physical device provides one ore more present rectangles for the display's surface
  // e.g. with two physical displays attached to a single GPU and a 1x2 Mosaic configuration spanning those displays,
  // that physical device will provide two present rectangles - one for each physical display - on the Mosaic's display
  // surface
  // the following code will combine these present rectangles into a single one to ease the rendering
  FramebufferRegions         framebufferRegions;
  vk::PhysicalDevice         physicalDevice = m_logicalDevice.getPhysicalDevice(deviceIndex);
  std::vector<vk::Rect2D>    presentRects   = physicalDevice.getPresentRectanglesKHR(m_surface.get());
  vk::SurfaceCapabilitiesKHR surfCaps       = physicalDevice.getSurfaceCapabilitiesKHR(m_surface.get());
  int32_t                    minX           = std::numeric_limits<int32_t>::max();
  int32_t                    maxX           = std::numeric_limits<int32_t>::min();
  int32_t                    minY           = std::numeric_limits<int32_t>::max();
  int32_t                    maxY           = std::numeric_limits<int32_t>::min();
  for(vk::Rect2D presentRect : presentRects)
  {
    framebufferRegions.emplace_back(FramebufferRegion{presentRect});
    minX = std::min(minX, presentRect.offset.x);
    maxX = std::max(maxX, presentRect.offset.x + (int32_t)presentRect.extent.width);
    minY = std::min(minY, presentRect.offset.y);
    maxY = std::max(maxY, presentRect.offset.y + (int32_t)presentRect.extent.height);
  }
  m_framebufferRegions.emplace(deviceIndex, std::move(framebufferRegions));

  // check if combined present rectangle contains pixels that are not contained in any individual present rectangle
  bool tightlyPacked = true;
  for(int32_t y = minY; tightlyPacked && y < maxY; ++y)
  {
    for(int32_t x = minX; tightlyPacked && x < minX; ++x)
    {
      auto coveredBy = std::find_if(presentRects.begin(), presentRects.end(),
                                    [=](vk::Rect2D const& presentRect) { return contains(presentRect, x, y); });
      tightlyPacked &= coveredBy != presentRects.end();
    }
  }
  char const* displayName = "unknown";
  for(vk::DisplayPropertiesKHR dispProps : physicalDevice.getDisplayPropertiesKHR())
  {
    if(dispProps.display == m_display)
    {
      displayName = dispProps.displayName;
    }
  }
  if(tightlyPacked)
  {
    LOGI("%d default present rectangle(s) of device %s on display %s were tightly packed into a single one.\n",
         presentRects.size(), formatVkDeviceName(physicalDevice).c_str(), displayName);
  }
  else
  {
    LOGW(
        "%d default present rectangles of device %s on display %s cannot be packed tightly. For optimal performance "
        "and correct results you may want to adjust your display configuration.\n",
        presentRects.size(), formatVkDeviceName(physicalDevice).c_str(), displayName);
  }

  // calculate the actual viewport from the surface's extent and its location on the canvas
  float        vpWidth   = (float)surfCaps.currentExtent.width / m_displayRegionOnCanvas.m_width;
  float        vpHeight  = (float)surfCaps.currentExtent.height / m_displayRegionOnCanvas.m_height;
  float        vpOffsetX = -vpWidth * m_displayRegionOnCanvas.m_offsetX;
  float        vpOffsetY = -vpHeight * m_displayRegionOnCanvas.m_offsetY;
  vk::Viewport viewport(vpOffsetX, vpOffsetY, vpWidth, vpHeight, 0.0f, 1.0f);
  vk::Rect2D   renderArea = vk::Rect2D{{minX, minY}, {(uint32_t)(maxX - minX), (uint32_t)(maxY - minY)}};
  m_canvasRegionsRenderThreads.emplace_back(
      std::make_unique<CanvasRegionRenderThread>(scene, m_logicalDevice, deviceIndex, renderArea, viewport));
  m_deviceMask.add(deviceIndex);
}

vk::PhysicalDevice LogicalDisplay::findMainPhysicalDevice() const
{
  for(DeviceIndex i = 0; i < m_logicalDevice.getNumPhysicalDevices(); ++i)
  {
    if(m_logicalDevice.getPhysicalDevice(i).getSurfaceSupportKHR(m_logicalDevice.getGraphicsQueueFamilyIndex(),
                                                                 m_surface.get())
       != 0)
    {
      return m_logicalDevice.getPhysicalDevice(i);
    }
  }
  return {};
}

void LogicalDisplay::querySurfaceFormats(std::vector<vk::SurfaceFormatKHR>& formats) const
{
  vk::PhysicalDevice dev = this->findMainPhysicalDevice();
  if(dev)
  {
    formats = dev.getSurfaceFormatsKHR(m_surface.get());
  }
}

bool LogicalDisplay::start(vk::SurfaceFormatKHR swapchainSurfFormat, vk::RenderPass renderPass)
{
  vk::PhysicalDevice mainPhysicalDevice = this->findMainPhysicalDevice();
  if(!mainPhysicalDevice)
  {
    LOGE("No physical device with display surface support found.\n");
    return false;
  }
  // vk_ddisplay
  // creating the swap chain images of the display surface is no different to conventional window surface swap chain
  // images creation
  // Note however that eFifo is currently the only supported present mode
  std::vector<vk::PresentModeKHR> surfPresentModes = mainPhysicalDevice.getSurfacePresentModesKHR(m_surface.get());
  if(surfPresentModes.empty())
  {
    LOGE("No present modes avaiable for display.\n");
    return false;
  }
  vk::PresentModeKHR presentMode = surfPresentModes.front();
  for(vk::PresentModeKHR pm : surfPresentModes)
  {
    if(pm == vk::PresentModeKHR::eFifo)
    {
      presentMode = pm;
      break;
    }
  }

  vk::SurfaceCapabilitiesKHR surfCaps = mainPhysicalDevice.getSurfaceCapabilitiesKHR(m_surface.get());
  uint32_t imageCount = std::max(surfCaps.minImageCount, std::min(NUM_QUEUED_FRAMES, surfCaps.maxImageCount));

  vk::DeviceGroupSwapchainCreateInfoKHR deviceGroupSwapchainCreateInfo(vk::DeviceGroupPresentModeFlagBitsKHR::eLocalMultiDevice);
  vk::SwapchainCreateInfoKHR swapchainCreateInfo(
      {}, m_surface.get(), imageCount, swapchainSurfFormat.format, swapchainSurfFormat.colorSpace, surfCaps.currentExtent,
      1, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, {},
      surfCaps.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque, presentMode, true, {}, &deviceGroupSwapchainCreateInfo);
  m_swapchain = m_logicalDevice.vkDevice().createSwapchainKHRUnique(swapchainCreateInfo);
  for(vk::UniqueSemaphore& sem : m_imageAcquiredSemaphores)
  {
    sem = m_logicalDevice.vkDevice().createSemaphoreUnique({});
  }
  m_readyToPresentSem = m_logicalDevice.vkDevice().createSemaphoreUnique({});

  vk::ImageCreateInfo depthStencilImageCreateInfo({}, vk::ImageType::e2D, vk::Format::eD24UnormS8Uint,
                                                  vk::Extent3D(surfCaps.currentExtent, 1), 1, 1, vk::SampleCountFlagBits::e1,
                                                  vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
                                                  vk::SharingMode::eExclusive, nullptr, vk::ImageLayout::eUndefined);
  m_depthStencil = m_logicalDevice.allocateImage({}, depthStencilImageCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);

  vk::ImageViewCreateInfo depthStencilImageViewCreateInfo(
      {}, m_depthStencil.m_image.get(), vk::ImageViewType::e2D, depthStencilImageCreateInfo.format, {},
      {vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1});
  m_depthStencilImageView = m_logicalDevice.vkDevice().createImageViewUnique(depthStencilImageViewCreateInfo);

  for(vk::Image swapchainImage : m_logicalDevice.vkDevice().getSwapchainImagesKHR(m_swapchain.get()))
  {
    vk::ImageViewCreateInfo swapchainImageViewCreateInfo({}, swapchainImage, vk::ImageViewType::e2D, swapchainSurfFormat.format,
                                                         {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
                                                          vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
                                                         vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    m_swapchainImageViews.emplace_back(m_logicalDevice.vkDevice().createImageViewUnique(swapchainImageViewCreateInfo));
    std::vector<vk::ImageView> framebufferAttachments = {m_swapchainImageViews.back().get(), m_depthStencilImageView.get()};
    vk::FramebufferCreateInfo frameBufferCreateInfo({}, renderPass, framebufferAttachments,
                                                    surfCaps.currentExtent.width, surfCaps.currentExtent.height, 1);
    m_framebuffers.emplace_back(m_logicalDevice.vkDevice().createFramebufferUnique(frameBufferCreateInfo));
  }

  for(auto& it : m_framebufferRegions)
  {
    DeviceIndex devIdx = it.first;
    for(FramebufferRegion& region : it.second)
    {
      vk::ImageCreateInfo createInfo({}, vk::ImageType::e2D, swapchainSurfFormat.format, vk::Extent3D{region.m_region.extent, 1},
                                     1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
                                     vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined);
      region.m_intermediate = m_logicalDevice.allocateImage(devIdx, createInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }
  }
  vk::DeviceSize sizePerPixel;
  switch(swapchainSurfFormat.format)
  {
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eR8G8B8A8Unorm:
      sizePerPixel = 4;
      break;
    default:
      LOGE("TODO");
#if defined(WIN32) && !defined(NDEBUG)
      _CrtDbgBreak();
#endif
      return false;
  }
  vk::BufferCreateInfo hostFramebufferCreateInfo({}, surfCaps.currentExtent.width * surfCaps.currentExtent.height * sizePerPixel,
                                                 vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive);
  m_hostFramebufferCopy = m_logicalDevice.allocateStagingBuffer(hostFramebufferCreateInfo);

  for(UniqueCanvasRegionRenderThread const& rt : m_canvasRegionsRenderThreads)
  {
    rt->start();
  }
  return true;
}

void LogicalDisplay::renderFrameAsync(CommandExecutionUnit& cmdExecUnit)
{
  // first the next swap chain image is acquired
  vk::Semaphore imageAcquiredSemaphore =
      m_imageAcquiredSemaphores[m_logicalDevice.getCurrentFrameIndex() % NUM_QUEUED_FRAMES].get();
  vk::AcquireNextImageInfoKHR acquireNextImageInfo(m_swapchain.get(), std::numeric_limits<uint64_t>::max(),
                                                   imageAcquiredSemaphore, {}, m_deviceMask);
  vk::ResultValue             rv = m_logicalDevice.vkDevice().acquireNextImage2KHR(acquireNextImageInfo);
  if(rv.result != vk::Result::eSuccess)
  {
    LOGE("acquireNextImage2KHR() failed.\n");
    return;
  }
  m_lastAcquiredSwapchainImageIdx = rv.value;
  m_lastAcquiredSwapchainImage =
      m_logicalDevice.vkDevice().getSwapchainImagesKHR(m_swapchain.get())[m_lastAcquiredSwapchainImageIdx];

  // the pre render cmd buffer will wait for the swap chain image's semaphore, transition the image to the
  // eColorAttachmentOptimal layout, and then notify each render context's individual semaphore
  // this must be done because one might have multiple threads rendering to the same image but a binary semaphore can
  // only be waited on a single time and the layout transition too must be executed only once
  m_preRenderCmdBuffer = cmdExecUnit.requestCommandBuffer(m_logicalDevice.getGraphicsQueueFamilyIndex());
  cmdExecUnit.pushWait(m_preRenderCmdBuffer, {imageAcquiredSemaphore, 0, vk::PipelineStageFlagBits2::eColorAttachmentOutput});
  for(UniqueCanvasRegionRenderThread const& rt : m_canvasRegionsRenderThreads)
  {
    rt->recordCommandsAsync(cmdExecUnit, m_framebuffers[m_lastAcquiredSwapchainImageIdx].get());
    cmdExecUnit.pushSignal(m_preRenderCmdBuffer,
                           {rt->getImageAcquiredSemaphore(), 0, vk::PipelineStageFlagBits2::eEarlyFragmentTests, 0});
  }
  std::vector<vk::ImageMemoryBarrier2> initialImageBarriers = {
      {vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
       vk::AccessFlagBits2::eMemoryWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
       m_logicalDevice.getGraphicsQueueFamilyIndex(), m_logicalDevice.getGraphicsQueueFamilyIndex(),
       m_lastAcquiredSwapchainImage, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)}};
  std::vector<vk::ImageMemoryBarrier2> temp;
  if(m_logicalDevice.getCurrentFrameIndex() == 0)
  {
    initialImageBarriers.emplace_back(
        vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eEarlyFragmentTests,
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal, m_logicalDevice.getGraphicsQueueFamilyIndex(),
        m_logicalDevice.getGraphicsQueueFamilyIndex(), m_depthStencil.m_image.get(),
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1));
  }
  m_preRenderCmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  m_preRenderCmdBuffer.pipelineBarrier2({vk::DependencyFlagBits::eByRegion, {}, {}, initialImageBarriers});
  m_preRenderCmdBuffer.end();
}

std::optional<LogicalDisplay::PresentData> LogicalDisplay::finishFrameRendering(CommandExecutionUnit& cmdExecUnit)
{
  for(UniqueCanvasRegionRenderThread const& rt : m_canvasRegionsRenderThreads)
  {
    rt->finishCommandRecording();
  }
  // the post render cmd buffer will wait for all render contexts to finish rendering, transition the swap chain image
  // to the present layout, and signal the present semaphore
  // in order to show a preview image in the control window, the swap chain image might be transfered to a separate
  // buffer from where it will be asynchronously processed further
  vk::CommandBuffer postRenderCmdBuffer = cmdExecUnit.requestCommandBuffer(m_logicalDevice.getGraphicsQueueFamilyIndex());
  for(UniqueCanvasRegionRenderThread const& rt : m_canvasRegionsRenderThreads)
  {
    cmdExecUnit.pushWait(postRenderCmdBuffer, {rt->getRenderDoneSemaphore(), 0, vk::PipelineStageFlagBits2::eAllCommands, 0});
  }
  postRenderCmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  // if(framebufferTransferQueueFamilyIdx.has_value())
  // {
  //   //this->storeFramebuffer(cmdExecUnit, framebufferTransferQueueFamilyIdx.value());
  //   //@todo
  // }
  // else
  {
    vk::ImageMemoryBarrier2 finalImageBarrier{vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                              vk::AccessFlagBits2::eMemoryWrite,
                                              vk::PipelineStageFlagBits2::eAllCommands,
                                              vk::AccessFlagBits2::eNone,
                                              vk::ImageLayout::eColorAttachmentOptimal,
                                              vk::ImageLayout::ePresentSrcKHR,
                                              m_logicalDevice.getGraphicsQueueFamilyIndex(),
                                              m_logicalDevice.getGraphicsQueueFamilyIndex(),
                                              m_lastAcquiredSwapchainImage,
                                              vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)};
    postRenderCmdBuffer.pipelineBarrier2({vk::DependencyFlagBits::eByRegion, {}, {}, finalImageBarrier});
  }
  postRenderCmdBuffer.end();
  cmdExecUnit.pushSignal(postRenderCmdBuffer, {m_readyToPresentSem.get(), 0, vk::PipelineStageFlagBits2::eAllCommands, 0});

  return PresentData{m_readyToPresentSem.get(), m_swapchain.get(), m_lastAcquiredSwapchainImageIdx};
}

void LogicalDisplay::storeFramebuffer(CommandExecutionUnit const& cmdExecUnit, uint32_t transferQueueFamilyIdx) {}

void LogicalDisplay::copyFramebufferToHost(vk::CommandBuffer cmdBuffer, vk::Buffer dstBuffer) {}

void LogicalDisplay::interrupt()
{
  for(UniqueCanvasRegionRenderThread const& rt : m_canvasRegionsRenderThreads)
  {
    rt->interrupt();
  }
}

void LogicalDisplay::join()
{
  for(UniqueCanvasRegionRenderThread const& rt : m_canvasRegionsRenderThreads)
  {
    rt->join();
  }
}
}  // namespace vkdd