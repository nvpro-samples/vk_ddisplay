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

#pragma once
#include "vkdd.hpp"

#include "buffer_allocation.hpp"
#include "canvas_region.hpp"
#include "image_allocation.hpp"
#include "vulkan_memory_pool.hpp"


namespace vkdd {
// vk_ddisplay
// a logical display repesents a single vkDisplay and manages the physical device independent resources, e.g. its
// surface, swap chain images, depth stencil image, etc
// additionally it manages all render contexts/threads that render to its display
class LogicalDisplay
{
public:
  typedef std::unique_ptr<class CanvasRegionRenderThread> UniqueCanvasRegionRenderThread;

  struct PresentData
  {
    vk::Semaphore    m_waitSem;
    vk::SwapchainKHR m_swapchain;
    uint32_t         m_imageIndex;
  };

  LogicalDisplay(class LogicalDevice& logicalDevice, vk::DisplayKHR display, CanvasRegion displayRegionOnCanvas);
  ~LogicalDisplay();

  [[nodiscard]] bool init(class Scene const& scene, std::vector<DeviceIndex> const& deviceIndices);
  vk::DisplayKHR     getDisplay() const { return m_display; }
  vk::SwapchainKHR   getSwapchain() const { return m_swapchain.get(); }
  DeviceMask const&  getDeviceMask() const { return m_deviceMask; }
  void               querySurfaceFormats(std::vector<vk::SurfaceFormatKHR>& formats) const;

  [[nodiscard]] bool         start(vk::SurfaceFormatKHR swapchainSurfFormat, vk::RenderPass renderPass);
  void                       renderFrameAsync(class CommandExecutionUnit& cmdExecUnit);
  std::optional<PresentData> finishFrameRendering(CommandExecutionUnit& cmdExecUnit);
  void                       copyFramebufferToHost(vk::CommandBuffer cmdBuffer, vk::Buffer dstBuffer);

  uint32_t                  getNumRenderThreads() const { return (uint32_t)m_canvasRegionsRenderThreads.size(); }
  CanvasRegionRenderThread* getRenderThread(uint32_t index) const { return m_canvasRegionsRenderThreads[index].get(); }

  void interrupt();
  void join();

private:
  struct FramebufferRegion
  {
    vk::Rect2D      m_region;
    ImageAllocation m_intermediate;
  };

  typedef std::vector<FramebufferRegion> FramebufferRegions;

  vk::DisplayKHR                                      m_display;
  CanvasRegion                                        m_displayRegionOnCanvas;
  LogicalDevice&                                      m_logicalDevice;
  std::vector<UniqueCanvasRegionRenderThread>         m_canvasRegionsRenderThreads;
  DeviceMask                                          m_deviceMask;
  vk::Extent2D                                        m_surfaceSize;
  vk::UniqueSurfaceKHR                                m_surface;
  vk::UniqueSwapchainKHR                              m_swapchain;
  std::array<vk::UniqueSemaphore, NUM_QUEUED_FRAMES>  m_imageAcquiredSemaphores;
  vk::CommandBuffer                                   m_preRenderCmdBuffer;
  vk::UniqueSemaphore                                 m_readyToPresentSem;
  std::vector<vk::UniqueImageView>                    m_swapchainImageViews;
  std::vector<vk::UniqueFramebuffer>                  m_framebuffers;
  ImageAllocation                                     m_depthStencil;
  vk::UniqueImageView                                 m_depthStencilImageView;
  uint32_t                                            m_lastAcquiredSwapchainImageIdx;
  vk::Image                                           m_lastAcquiredSwapchainImage;
  std::unordered_map<DeviceIndex, FramebufferRegions> m_framebufferRegions;
  BufferAllocation                                    m_hostFramebufferCopy;

  vk::PhysicalDevice findMainPhysicalDevice() const;
  void               pushRenderContext(Scene const& scene, DeviceIndex deviceIndex);
  void               storeFramebuffer(CommandExecutionUnit const& cmdExecUnit, uint32_t transferQueueFamilyIdx);
};
}  // namespace vkdd