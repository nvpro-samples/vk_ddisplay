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

#include "scene.hpp"

#include <nvgl/appwindowprofiler_gl.hpp>
#include <unordered_set>

namespace vkdd {
class VkDDisplayApp : public nvgl::AppWindowProfilerGL
{
public:
  VkDDisplayApp(vk::UniqueInstance instance);
  ~VkDDisplayApp();

  bool begin() override;
  void end() override;
  void think(double time) override;

private:
  struct DisplayInfo
  {
    vk::DisplayPropertiesKHR     m_props;
    uint32_t                     m_deviceGroupIndex;
    std::unordered_set<uint32_t> m_physicalDeviceIndices;
  };

  std::string                                                                    m_configPath;
  std::vector<DisplayInfo>                                                       m_displayInfos;
  Scene                                                                          m_scene;
  vk::UniqueInstance                                                             m_instance;
  std::unordered_map<uint32_t, std::unique_ptr<class LogicalDevice>>             m_logicalDevices;
  bool                                                                           m_paused = false;
  std::vector<std::pair<class LogicalDisplay*, class CanvasRegionRenderThread*>> m_possibleSelections;
  uint32_t                                                                       m_activeSelectionIndex;

  void           queryTolopogy();
  LogicalDevice* getLogicalDevice(uint32_t devGroupIdx);
  void           visitSelection(std::function<void(CanvasRegionRenderThread*)> visitor);
  void           setActiveSelection(uint32_t activeSelectionIndex);
  void           renderGui();
  void           handleInput();
  bool           enableDisplay(uint32_t globalDisplayIndex, struct CanvasRegion canvasRegion);
  bool           parseDDisplayConfig();
};
}  // namespace vkdd
