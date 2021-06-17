#pragma once

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Optional.h>
#include <Corrade/PluginManager/Manager.h>
#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/DebugStl.h>

#include <Magnum/GL/Buffer.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/MeshTools/Interleave.h>
#include <Magnum/MeshTools/CompressIndices.h>
#include <Magnum/Primitives/Cube.h>

#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/Shaders/VertexColor.h>

#include <Magnum/ImageView.h>
#include <Magnum/Mesh.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Math/Color.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/Trade/MeshObjectData3D.h>
#include <Magnum/Trade/PhongMaterialData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/TextureData.h>

#include <Magnum/Primitives/Icosphere.h>

#include "Arktos/Log.h"

#include <spdlog/spdlog.h>

using namespace Magnum;
using namespace Math::Literals;

typedef Magnum::SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
typedef Magnum::SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;

namespace Arktos {
    class BaseApplication : public Magnum::Platform::Application {
    public:
        explicit BaseApplication(const Arguments& arguments);

        void drawEvent() override;

        void viewportEvent(ViewportEvent& event) override;

        void keyPressEvent(KeyEvent& event) override;
        void keyReleaseEvent(KeyEvent& event) override;

        void mousePressEvent(MouseEvent& event) override;
        void mouseReleaseEvent(MouseEvent& event) override;
        void mouseMoveEvent(MouseMoveEvent& event) override;
        void mouseScrollEvent(MouseScrollEvent& event) override;

        void textInputEvent(TextInputEvent& event) override;

        Magnum::Vector3 positionOnSphere(const Magnum::Vector2i& position) const;
        void addObject(Trade::AbstractImporter& importer, Containers::ArrayView<const Containers::Optional<Trade::PhongMaterialData>> materials, Object3D& parent, UnsignedInt i);

    private:
        Magnum::ImGuiIntegration::Context _imgui{Magnum::NoCreate};

        bool _showDemoWindow = true;
        bool _showAnotherWindow = false;
        Color4 _clearColor = 0x72909aff_rgbaf;
        Magnum::Float _floatValue = 1.0f;
        Magnum::Math::Vector3<Magnum::Float> _cubePos = {0.0f, 0.0f, 0.0f};

        Magnum::GL::Mesh _mesh;
        Magnum::Shaders::Phong _shader;

        Matrix4 _transformation, _projection, _rotation;
        Color3 _color;

        Shaders::Phong _coloredShader, _texturedShader;
    };
}