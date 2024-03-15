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

namespace vkdd {
class Scene
{
public:
  enum class NodeType
  {
    TORUS,
    //SPHERE,
  };

  struct Node
  {
  public:
    Node(NodeType nodeType, Vec3f scaling, Angle roll, Angle pitch, Angle yaw, Vec3f translation);

    uint32_t getId() const { return m_id; }
    NodeType getNodeType() const { return m_nodeType; }
    void     setRotation(Angle roll, Angle pitch, Angle yaw);
    Mat4x4f  createModel() const;

  private:
    uint32_t m_id;
    NodeType m_nodeType;
    Vec3f    m_scaling;
    Angle    m_roll;
    Angle    m_pitch;
    Angle    m_yaw;
    Vec3f    m_translation;
  };

  struct PerspectiveCamera
  {
    Vec3f   m_pos;
    Angle   m_fov;
    float   m_aspect;
    float   m_nearZ;
    float   m_farZ;
    Mat4x4f m_view;
    Mat4x4f m_proj;
  };

  Scene();

  void update(float millis);
  void collectVisibleNodes(vk::Viewport globalViewport, vk::Viewport localViewport, std::function<void(Node const& node)> onVisible) const;
  PerspectiveCamera const& getCamera() const { return m_camera; }
  void                     setPerspectiveCamera(float aspect, Angle fov, float nearZ, float farZ);
  float                    getRuntimeMillis() const { return m_runtimeMillis; }
  uint64_t                 getNumUpdates() const { return m_numUpdates; }
  int32_t&                 getDesiredNumDonutsX() { return m_desiredNumDonutsX; }
  int32_t&                 getDesiredNumDonutsY() { return m_desiredNumDonutsY; }
  void                     increaseNumDonutsX();
  void                     decreaseNumDonutsX();
  void                     increaseNumDonutsY();
  void                     decreaseNumDonutsY();

private:
  uint64_t          m_numUpdates    = 0;
  float             m_runtimeMillis = 0.0f;
  PerspectiveCamera m_camera;
  int32_t           m_desiredNumDonutsX;
  int32_t           m_desiredNumDonutsY;
  int32_t           m_numDonutsX;
  int32_t           m_numDonutsY;
  std::vector<Node> m_geometryNodes;

  void rebuild();
  void fillDonutPlane(float z, uint32_t numDonutsX, uint32_t numDonutsY);
};
}  // namespace vkdd
