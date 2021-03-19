#pragma once

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/Math/Color.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Shaders/VertexColor.h>

#include "Arktos/Log.h"

#include <spdlog/spdlog.h>

using namespace Magnum;
using namespace Math::Literals;

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

    private:
        Magnum::ImGuiIntegration::Context _imgui{Magnum::NoCreate};

        bool _showDemoWindow = true;
        bool _showAnotherWindow = false;
        Color4 _clearColor = 0x72909aff_rgbaf;
        Magnum::Float _floatValue = 0.0f;

        Magnum::GL::Mesh _mesh;
        Magnum::Shaders::VertexColor2D _shader;
    };
}