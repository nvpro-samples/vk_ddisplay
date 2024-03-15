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

#include "render_thread.hpp"

namespace vkdd {
class CanvasRegionRenderThread : public RenderThread
{
public:
  CanvasRegionRenderThread(class Scene const&   scene,
                           class LogicalDevice& logicalDevice,
                           DeviceIndex          deviceIndex,
                           vk::Rect2D           renderArea,
                           vk::Viewport         viewport);

  void     recordCommands(class CommandExecutionUnit& cmdExecUnit, vk::Framebuffer framebuffer) override;
  void     incNumFurLayers() { ++m_numFurLayers; }
  void     decNumFurLayers() { m_numFurLayers = std::max(1, m_numFurLayers - 1); }
  int32_t& getNumFurLayers() { return m_numFurLayers; }
  void     setHighlighted(bool highlighted) { m_highlighted = highlighted; }
  Vec3f    getLastClearColor() const { return m_lastClearColor; }

private:
  Scene const&                                   m_scene;
  vk::Rect2D                                     m_renderArea;
  vk::Viewport                                   m_viewport;
  int32_t                                        m_numFurLayers = 32;
  std::unique_ptr<class TriangleMeshInstanceSet> m_instances;
  vk::UniqueSemaphore                            m_syncTimelineSemaphore;
  uint64_t                                       m_syncTimelineSemaphoreValue;
  bool                                           m_highlighted;
  Vec3f                                          m_lastClearColor;
};
}  // namespace vkdd
