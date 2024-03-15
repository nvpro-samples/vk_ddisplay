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

#include "vk_ddisplay_app.hpp"

#include "canvas_region_render_thread.hpp"
#include "logical_device.hpp"
#include "logical_display.hpp"

#include <backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_gl.h>
#include <imgui/imgui_helper.h>
#include <json.hpp>

#include <fstream>

namespace vkdd {
VkDDisplayApp::VkDDisplayApp(vk::UniqueInstance instance)
    : m_instance(std::move(instance))
    , m_possibleSelections{std::make_pair(nullptr, nullptr)}
    , m_activeSelectionIndex(0)
{
  m_parameterList.add("config|Path to the json file containing the ddisplay configuration", &m_configPath);
  m_parameterList.add("topology-only|If set, the app closes automatically after printing the system's topology",
                      [](uint32_t t) { exit(0); });
  this->queryTolopogy();
  this->setVsync(false);
}

VkDDisplayApp::~VkDDisplayApp() {}

void VkDDisplayApp::queryTolopogy()
{
  // vk_ddisplay
  // first we will collect the system topology
  // nothe that in case of an active Mosaic configuration across multiple GPUs, the same vkDisplay will show up on multiple vkPhysicalDevices
  std::vector<vk::PhysicalDeviceGroupProperties> devGroups = m_instance->enumeratePhysicalDeviceGroups();
  LOGI(
      "--------------------------------------------------------------------------------\n"
      "System topology:\n");
  for(uint32_t devGroupIdx = 0; devGroupIdx < devGroups.size(); ++devGroupIdx)
  {
    LOGI("device group [%d]\n", devGroupIdx);
    for(uint32_t devIdx = 0; devIdx < devGroups[devGroupIdx].physicalDeviceCount; ++devIdx)
    {
      vk::PhysicalDevice physicalDevice = devGroups[devGroupIdx].physicalDevices[devIdx];
      LOGI(" physical device [%d]: %s\n", devIdx, formatVkDeviceName(physicalDevice).c_str());
      auto physicalDeviceDisplays = physicalDevice.getDisplayPropertiesKHR();
      for(uint32_t dispIdx = 0; dispIdx < physicalDeviceDisplays.size(); ++dispIdx)
      {
        vk::DisplayPropertiesKHR dispProps = physicalDeviceDisplays[dispIdx];
        auto displayInfoPred = [&](DisplayInfo const& di) { return di.m_props.display == dispProps.display; };
        if(auto findIt = std::find_if(m_displayInfos.begin(), m_displayInfos.end(), displayInfoPred);
           findIt != m_displayInfos.end())
        {
          findIt->m_physicalDeviceIndices.insert(devIdx);
        }
        else
        {
          m_displayInfos.emplace_back(DisplayInfo{dispProps, devGroupIdx, {devIdx}});
        }

        LOGI("  display [%d]: %s; supported display modes: ", dispIdx, dispProps.displayName);
        char const* sep = "";
        for(vk::DisplayModePropertiesKHR modeProps : physicalDevice.getDisplayModePropertiesKHR(dispProps.display))
        {
          LOGI("%s%d x %d @ %.3f Hz", sep, modeProps.parameters.visibleRegion.width,
               modeProps.parameters.visibleRegion.height, 1e-3f * (float)modeProps.parameters.refreshRate);
          sep = ", ";
        }
        LOGI("\n");
      }
    }
  }
  LOGI(
      "--------------------------------------------------------------------------------\n"
      "Usable displays:\n");
  for(uint32_t i = 0; i < m_displayInfos.size(); ++i)
  {
    LOGI("[%d] %s: attached to physical device(s) { ", i, m_displayInfos[i].m_props.displayName);
    char const* sep = "";
    for(uint32_t devIdx : m_displayInfos[i].m_physicalDeviceIndices)
    {
      LOGI("%s%d", sep, devIdx);
      sep = ", ";
    }
    LOGI(" } of device group %d.\n", m_displayInfos[i].m_deviceGroupIndex);
  }
  if(m_displayInfos.empty())
  {
    LOGE("None\n");
  }
  LOGI("--------------------------------------------------------------------------------\n");
}

bool VkDDisplayApp::begin()
{
  ImGuiH::Init(this->getWidth(), this->getHeight(), this, ImGuiH::FONT_MONOSPACED_SCALED);
  if(!ImGui_ImplGlfw_InitForOpenGL(m_internal, true))
  {
    LOGE("ImGui_ImplGlfw_InitForOpenGL() failed.\n");
    return false;
  }
  ImGui::InitGL();

  if(!m_configPath.empty())
  {
    if(!this->parseDDisplayConfig())
    {
      return false;
    }
  }
  else if(!this->enableDisplay(0, {}))
  {
    LOGE("Default configuration failed.\n");
    return false;
  }

  // vk_ddisplay
  // when everything is set up, the rendering can be started
  if(m_logicalDevices.empty())
  {
    LOGE("No displays were enabled.\n");
    return false;
  }
  for(auto const& logicalDevicesIt : m_logicalDevices)
  {
    if(!logicalDevicesIt.second->start())
    {
      LOGE("Failed to start logical device.");
      return false;
    }
  }
  return true;
}

bool VkDDisplayApp::enableDisplay(uint32_t globalDisplayIndex, CanvasRegion canvasRegion)
{
  if(m_displayInfos.size() <= globalDisplayIndex)
  {
    LOGE("Display index (%d) must be positive and less than the number of displays (%d).\n", globalDisplayIndex,
         m_displayInfos.size());
    return false;
  }
  else
  {
    // vk_ddisplay
    // at this point an individual logical display will be activated
    // one needs to find the logical device (which represents a Vulkan device group) to which the display is connected and enable it
    DisplayInfo const& dispInfo = m_displayInfos[globalDisplayIndex];
    LogicalDisplay*    logicalDisplay =
        this->getLogicalDevice(dispInfo.m_deviceGroupIndex)->enableDisplay(m_scene, dispInfo.m_props.display, canvasRegion);
    if(logicalDisplay)
    {
      m_possibleSelections.emplace_back(std::make_pair(logicalDisplay, nullptr));
      for(uint32_t i = 0; i < logicalDisplay->getNumRenderThreads(); ++i)
      {
        m_possibleSelections.emplace_back(std::make_pair(logicalDisplay, logicalDisplay->getRenderThread(i)));
      }
      return true;
    }
    else
    {
      LOGE("Failed to enable display %d.\n", globalDisplayIndex);
      return false;
    }
  }
}

void VkDDisplayApp::think(double time)
{
  this->handleInput();
  static double lastTime = time;
  if(!m_paused)
  {
    float frameTimeMillis = 1e3f * (time - lastTime);
    m_scene.update(frameTimeMillis);
    for(auto& logicalDeviceIt : m_logicalDevices)
    {
      logicalDeviceIt.second->render();
    }
  }
  this->renderGui();
  lastTime = time;
}

void VkDDisplayApp::renderGui()
{
  glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::SetNextWindowSize(ImGuiH::dpiScaled(480, 0), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowPos(ImGuiH::dpiScaled(20, 20), ImGuiCond_FirstUseEver);
  if(ImGui::Begin("Scene", 0, ImGuiWindowFlags_NoResize))
  {
    ImGui::Checkbox("Pause rendering", &m_paused);
    ImGui::SliderInt("Number of donuts X", &m_scene.getDesiredNumDonutsX(), 1, 48);
    ImGui::SliderInt("Number of donuts Y", &m_scene.getDesiredNumDonutsY(), 1, 48);
    ImGui::End();
  }

  int i = 0;
  for(std::pair<LogicalDisplay*, CanvasRegionRenderThread*> s : m_possibleSelections)
  {
    if(s.second)
    {
      std::stringstream title;
      title << "Render Thread " << i;
      ImGui::SetNextWindowSize(ImGuiH::dpiScaled(480, 0), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowPos(ImGuiH::dpiScaled(20, 120 + i * 70), ImGuiCond_FirstUseEver);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {14, 12});
      if(ImGui::Begin(title.str().c_str(), 0, ImGuiWindowFlags_NoResize))
      {
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        uint32_t    color    = IM_COL32((uint32_t)255.0f * s.second->getLastClearColor().x,
                                        (uint32_t)255.0f * s.second->getLastClearColor().y,
                                        (uint32_t)255.0f * s.second->getLastClearColor().z, 255);
        ImVec2      tl       = {ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMin().x - 8,
                                ImGui::GetWindowPos().y + ImGui::GetWindowContentRegionMin().y - 8};
        ImVec2      br1      = {ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x, tl.y + 6};
        ImVec2      br2      = {tl.x + 6, ImGui::GetWindowPos().y + ImGui::GetWindowContentRegionMax().y};
        drawList->AddRectFilled(tl, br1, color);
        drawList->AddRectFilled(tl, br2, color);
        ImGui::SliderInt("Fur layers", &s.second->getNumFurLayers(), 1, 128);

        ImGui::End();
      }
      ImGui::PopStyleVar();
      ++i;
    }
  }
  ImGui::Render();
  ImGui::RenderDrawDataGL(ImGui::GetDrawData());
  ImGui::EndFrame();
}

void VkDDisplayApp::end()
{
  for(auto& logicalDeviceIt : m_logicalDevices)
  {
    logicalDeviceIt.second->interrupt();
  }
  for(auto& logicalDeviceIt : m_logicalDevices)
  {
    logicalDeviceIt.second->join();
  }
}

LogicalDevice* VkDDisplayApp::getLogicalDevice(uint32_t devGroupIdx)
{
  if(auto findIt = m_logicalDevices.find(devGroupIdx); findIt != m_logicalDevices.end())
  {
    return findIt->second.get();
  }
  return m_logicalDevices.emplace(devGroupIdx, std::make_unique<LogicalDevice>(m_instance.get(), devGroupIdx))
      .first->second.get();
}

void VkDDisplayApp::setActiveSelection(uint32_t activeSelectionIndex)
{
  this->visitSelection([](CanvasRegionRenderThread* renderThread) { renderThread->setHighlighted(false); });
  m_activeSelectionIndex = (activeSelectionIndex % m_possibleSelections.size() + (uint32_t)m_possibleSelections.size())
                           % m_possibleSelections.size();
  if(m_activeSelectionIndex != 0)
  {
    this->visitSelection([](CanvasRegionRenderThread* renderThread) { renderThread->setHighlighted(true); });
  }
}

void VkDDisplayApp::visitSelection(std ::function<void(CanvasRegionRenderThread*)> visitor)
{
  std::pair<LogicalDisplay*, CanvasRegionRenderThread*> selection = m_possibleSelections[m_activeSelectionIndex];
  if(selection.second)
  {
    visitor(selection.second);
  }
  else if(selection.first)
  {
    for(uint32_t i = 0; i < selection.first->getNumRenderThreads(); ++i)
    {
      visitor(selection.first->getRenderThread(i));
    }
  }
  else
  {
    for(std::pair<LogicalDisplay*, CanvasRegionRenderThread*> s : m_possibleSelections)
    {
      if(s.second)
      {
        visitor(s.second);
      }
    }
  }
}

bool VkDDisplayApp::parseDDisplayConfig()
{
  std::ifstream configJson(m_configPath);
  if(!configJson)
  {
    LOGE("Failed to read config json file %s.\n", m_configPath.c_str());
    return false;
  }
  nlohmann::json config = nlohmann::json::parse(configJson);
  if(!config.contains("canvas") || !config["canvas"].is_object())
  {
    LOGE("The config must contain a \"canvas\" object entry.\n");
    return false;
  }
  if(!config["canvas"].contains("aspectNum") || !config["canvas"]["aspectNum"].is_number())
  {
    LOGE("The \"canvas\" object must define an \"aspectNum\" float value.\n");
    return false;
  }
  if(!config["canvas"].contains("aspectDen") || !config["canvas"]["aspectDen"].is_number())
  {
    LOGE("The \"canvas\" object must define an \"aspectDen\" float value.\n");
    return false;
  }
  if(!config["canvas"].contains("fov") || !config["canvas"]["fov"].is_number())
  {
    LOGE("The \"canvas\" object must define an \"fov\" float value.\n");
    return false;
  }
  m_scene.setPerspectiveCamera((float)config["canvas"]["aspectNum"] / (float)config["canvas"]["aspectDen"],
                               Angle::degree(config["canvas"]["fov"]), 1e-2f, 1e+2f);

  if(!config.contains("displays") || !config["displays"].is_array())
  {
    LOGE("The config must contain a \"displays\" array entry.\n");
    return false;
  }
  for(nlohmann::json const& displayJson : config["displays"])
  {
    uint32_t     dispIdx;
    CanvasRegion canvasRegion;
    if(displayJson.is_number_unsigned())
    {
      dispIdx = displayJson;
    }
    else if(displayJson.is_object())
    {
      if(!displayJson.contains("index") || !displayJson["index"].is_number_unsigned())
      {
        LOGE("Each object entry of \"displays\" must provide an positive \"index\" integer.\n");
        return false;
      }
      dispIdx = displayJson["index"];
      if(displayJson.contains("canvasOffsetX") && displayJson["canvasOffsetX"].is_number())
      {
        canvasRegion.m_offsetX = displayJson["canvasOffsetX"];
      }
      if(displayJson.contains("canvasOffsetY") && displayJson["canvasOffsetY"].is_number())
      {
        canvasRegion.m_offsetY = displayJson["canvasOffsetY"];
      }
      if(displayJson.contains("canvasWidth") && displayJson["canvasWidth"].is_number())
      {
        canvasRegion.m_width = displayJson["canvasWidth"];
      }
      if(displayJson.contains("canvasHeight") && displayJson["canvasHeight"].is_number())
      {
        canvasRegion.m_height = displayJson["canvasHeight"];
      }
    }
    else
    {
      LOGE("All entries of the the \"displays\" array must be single unsigned integers or objects.\n");
      return false;
    }
    this->enableDisplay(dispIdx, canvasRegion);
  }
  return true;
}

void VkDDisplayApp::handleInput()
{
  if(m_windowState.onPress(KeyCode::KEY_RIGHT))
  {
    m_scene.increaseNumDonutsX();
  }
  if(m_windowState.onPress(KeyCode::KEY_LEFT))
  {
    m_scene.decreaseNumDonutsX();
  }
  if(m_windowState.onPress(KeyCode::KEY_UP))
  {
    m_scene.increaseNumDonutsY();
  }
  if(m_windowState.onPress(KeyCode::KEY_DOWN))
  {
    m_scene.decreaseNumDonutsY();
  }
  if(m_windowState.onPress(KeyCode::KEY_SPACE))
  {
    m_paused = !m_paused;
  }
  if(m_windowState.onPress(KeyCode::KEY_PAGE_UP))
  {
    this->setActiveSelection(m_activeSelectionIndex + 1);
  }
  if(m_windowState.onPress(KeyCode::KEY_PAGE_DOWN))
  {
    this->setActiveSelection(m_activeSelectionIndex - 1);
  }
  if(m_windowState.onPress(KeyCode::KEY_KP_ADD))
  {
    this->visitSelection([](CanvasRegionRenderThread* renderThread) { renderThread->incNumFurLayers(); });
  }
  if(m_windowState.onPress(KeyCode::KEY_KP_SUBTRACT))
  {
    this->visitSelection([](CanvasRegionRenderThread* renderThread) { renderThread->decNumFurLayers(); });
  }
}
}  // namespace vkdd