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

#include "canvas_region_render_thread.hpp"

#include "command_execution_unit.hpp"
#include "logical_device.hpp"
#include "scene.hpp"
#include "triangle_mesh_instance_set.hpp"
#include "vulkan_memory_object_uploader.hpp"

namespace vkdd {
CanvasRegionRenderThread::CanvasRegionRenderThread(class Scene const& scene,
                                                   LogicalDevice&     logicalDevice,
                                                   DeviceIndex        deviceIndex,
                                                   vk::Rect2D         renderArea,
                                                   vk::Viewport       viewport)
    : RenderThread(logicalDevice, deviceIndex)
    , m_scene(scene)
    , m_renderArea(renderArea)
    , m_viewport(viewport)
    , m_instances(std::make_unique<TriangleMeshInstanceSet>(logicalDevice, deviceIndex))
    , m_highlighted(false)
{
}

void CanvasRegionRenderThread::recordCommands(class CommandExecutionUnit& cmdExecUnit, vk::Framebuffer framebuffer)
{
  std::array<Vec3f, 14> const COLORS = {Colors::STRONG_RED, Colors::GREEN_NV, Colors::BONDI_BLUE, Colors::RED,
                                        Colors::GREEN,      Colors::BLUE,     Colors::CYAN,       Colors::MAGENTA,
                                        Colors::YELLOW,     Colors::WHITE,    Colors::LIGHT_GRAY, Colors::GRAY,
                                        Colors::DARK_GRAY,  Colors::BLACK};

  m_lastClearColor = COLORS[this->getSystemPhysicalDeviceIndex() % sizeof(COLORS[0])];
  if(m_highlighted)
  {
    m_lastClearColor = lerp(m_lastClearColor, Colors::DARK_GRAY, 0.5f + 0.5f * std::sinf(1e-2f * m_scene.getRuntimeMillis()));
  }

  m_instances->beginInstanceCollection();
  m_scene.collectVisibleNodes({}, {}, [this](Scene::Node const& node) {
    // right now the app only supports torus geometry
    // in practice you would first want to check the torus' visibility in this render context before adding its
    // instances (we may add this in a later update)
    if(node.getNodeType() == Scene::NodeType::TORUS)
    {
      // for a simple fur effect the app renders the same geometry in multiple layers (or shells), where each additional
      // layer discards more fragments than the previous
      m_numFurLayers       = std::max(1, m_numFurLayers);
      float   maxExtrusion = 0.3f;
      Mat4x4f model        = node.createModel();
      for(int32_t i = 0; i < m_numFurLayers; ++i)
      {
        float shellHeight = (float)i / (float)m_numFurLayers;
        float extrusion   = maxExtrusion * shellHeight;
        m_instances->pushInstance(node.getId(), model, shellHeight, extrusion);
      }
    }
  });
  m_instances->endInstanceCollection();

  std::vector<uint32_t> queueFamilyIndices = {this->getLogicalDevice().getGraphicsQueueFamilyIndex()};
  if(m_instances->getNumInstances() != 0)
  {
    queueFamilyIndices.emplace_back(this->getLogicalDevice().getTransferQueueFamilyIndex());
  }
  std::vector<vk::CommandBuffer> cmdBuffers =
      cmdExecUnit.requestCommandBuffers(queueFamilyIndices, DeviceMask::ofSingleDevice(this->getDeviceIndex()));

  vk::CommandBuffer graphicsCmdBuffer = cmdBuffers.front();
  cmdExecUnit.pushWait(graphicsCmdBuffer, {this->getImageAcquiredSemaphore(), 0,
                                           vk::PipelineStageFlagBits2::eColorAttachmentOutput, this->getDeviceIndex()});
  graphicsCmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  if(m_instances->getNumInstances() != 0)
  {
    if(!m_syncTimelineSemaphore)
    {
      vk::SemaphoreTypeCreateInfo semType({vk::SemaphoreType::eTimeline});
      m_syncTimelineSemaphore      = this->getLogicalDevice().vkDevice().createSemaphoreUnique({{}, &semType});
      m_syncTimelineSemaphoreValue = 0;
    }
    GlobalData globalData      = {};
    globalData.m_view          = m_scene.getCamera().m_view;
    globalData.m_proj          = m_scene.getCamera().m_proj;
    globalData.m_runtimeMillis = m_scene.getRuntimeMillis();

    // the instance buffer will be updated through a dedicated transfer Vulkan queue which requires proper
    // synchronization and queu ownership transfers
    vk::CommandBuffer transferCmdBuffer = cmdBuffers.back();
    cmdExecUnit.pushWait(transferCmdBuffer, {m_syncTimelineSemaphore.get(), m_syncTimelineSemaphoreValue,
                                             vk::PipelineStageFlagBits2::eTransfer, this->getDeviceIndex()});
    transferCmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    m_instances->updateDeviceMemory(transferCmdBuffer, graphicsCmdBuffer);
    transferCmdBuffer.end();
    cmdExecUnit.pushSignal(transferCmdBuffer, {m_syncTimelineSemaphore.get(), ++m_syncTimelineSemaphoreValue,
                                               vk::PipelineStageFlagBits2::eTransfer, this->getDeviceIndex()});

    cmdExecUnit.pushWait(graphicsCmdBuffer, {m_syncTimelineSemaphore.get(), m_syncTimelineSemaphoreValue,
                                             vk::PipelineStageFlagBits2::eVertexAttributeInput, this->getDeviceIndex()});
    vk::ClearColorValue         clearColorValue(m_lastClearColor.x, m_lastClearColor.y, m_lastClearColor.z, 1.0f);
    vk::ClearDepthStencilValue  clearDepthStencil(1.0f, 0U);
    std::vector<vk::ClearValue> clearValues = {clearColorValue, clearDepthStencil};
    vk::RenderPassBeginInfo renderPassBegin(this->getLogicalDevice().getDonutRenderPass(), framebuffer, m_renderArea, clearValues);
    graphicsCmdBuffer.beginRenderPass(renderPassBegin, vk::SubpassContents::eInline);
    graphicsCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, this->getLogicalDevice().getDonutPipeline());
    graphicsCmdBuffer.pushConstants<GlobalData>(this->getLogicalDevice().getDonutPipelineLayout(),
                                                vk::ShaderStageFlagBits::eVertex, 0, globalData);
    TriangleMesh* donutTriMesh = this->getLogicalDevice().getDonutTriangleMesh(this->getDeviceIndex(), 16);
    // the vertex and index buffers of the triangle mesh might not be ready yet
    // in that case one has to synchronize with its timeline semaphore
    if(this->getLogicalDevice().getCurrentFrameIndex() < donutTriMesh->getAvailableFrameIndex())
    {
      cmdExecUnit.pushWait(graphicsCmdBuffer, {this->getLogicalDevice().getUploader().getSyncSemaphore(),
                                               this->getLogicalDevice().getCurrentFrameIndex() + 1,
                                               vk::PipelineStageFlagBits2::eVertexAttributeInput, this->getDeviceIndex()});
    }
    graphicsCmdBuffer.setViewport(0, m_viewport);

    // vk_ddisplay
    // one must ensure to only render to the parts of the surface which are covered by the physical device's present
    // rectangles. the easiest way to do this is by setting up the scissor rectangle(s) appropriately
    graphicsCmdBuffer.setScissor(0, m_renderArea);
    m_instances->draw(graphicsCmdBuffer, *donutTriMesh);
    graphicsCmdBuffer.endRenderPass();
    cmdExecUnit.pushSignal(graphicsCmdBuffer, {m_syncTimelineSemaphore.get(), ++m_syncTimelineSemaphoreValue,
                                               vk::PipelineStageFlagBits2::eVertexAttributeInput, this->getDeviceIndex()});
  }
  graphicsCmdBuffer.end();
  cmdExecUnit.pushSignal(graphicsCmdBuffer, {this->getRenderDoneSemaphore(), 0,
                                             vk::PipelineStageFlagBits2::eColorAttachmentOutput, this->getDeviceIndex()});
}
}  // namespace vkdd