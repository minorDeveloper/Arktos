#include "Arktos.h"

using namespace Magnum;
using namespace Math::Literals;

namespace Arktos {
    BaseApplication::BaseApplication(const Arguments& arguments):
            Platform::Application{arguments, Configuration{}
                    .setTitle("Arktos")}
    {
        Arktos::Log::Init();

        CORE_INFO("Hello this is a log message");

        _imgui = ImGuiIntegration::Context(Vector2{windowSize()} / dpiScaling(),
                                           windowSize(), framebufferSize());
        /* Imgui rendering */
        GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,
                                       GL::Renderer::BlendEquation::Add);
         GL::Renderer::setBlendFunction(GL::Renderer::BlendFunction::SourceAlpha,
                                        GL::Renderer::BlendFunction::OneMinusSourceAlpha);

        GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
        GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

        Trade::MeshData cube = Primitives::cubeSolid();

        Trade::MeshData icoSphere = Primitives::icosphereSolid(1);

        GL::Buffer vertices;
        vertices.setData(MeshTools::interleave(cube.positions3DAsArray(),
                                               cube.normalsAsArray()));

        std::pair<Containers::Array<char>, MeshIndexType> compressed =
                MeshTools::compressIndices(cube.indicesAsArray());
        GL::Buffer indices;
        indices.setData(compressed.first);

        /*_mesh.setPrimitive(cube.primitive())
                .setCount(cube.indexCount())
                .addVertexBuffer(std::move(vertices), 0, Shaders::Phong::Position{},
                                 Shaders::Phong::Normal{})
                .setIndexBuffer(std::move(indices), 0, compressed.second);
        */
        _mesh = MeshTools::compile(Primitives::icosphereSolid(4));

        _rotation = Matrix4::rotationX(30.0_degf)*Matrix4::rotationY(40.0_degf);
        _projection =
                Matrix4::perspectiveProjection(
                        35.0_degf, Vector2{windowSize()}.aspectRatio(), 0.01f, 100.0f)*
                Matrix4::translation(Vector3::zAxis(-10.0f));
        _color = Color3::fromHsv({35.0_degf, 1.0f, 1.0f});
    }

    void BaseApplication::drawEvent() {
        GL::defaultFramebuffer.clear(
                GL::FramebufferClear::Color|GL::FramebufferClear::Depth);

        _imgui.newFrame();

        /* Enable text input, if needed */
        if (ImGui::GetIO().WantTextInput && !isTextInputActive())
            startTextInput();
        else if (!ImGui::GetIO().WantTextInput && isTextInputActive())
            stopTextInput();

        /* 1. Show a simple window.
   Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appear in
   a window called "Debug" automatically */
        {
            ImGui::Text("Hello, world!");
            ImGui::SliderFloat("X", &_cubePos[0], -2.0f, 2.0f);
            ImGui::SliderFloat("Y", &_cubePos[1], -2.0f, 2.0f);
            ImGui::SliderFloat("Z", &_cubePos[2], -2.0f, 2.0f);
            if (ImGui::ColorEdit3("Clear Color", _clearColor.data()))
                GL::Renderer::setClearColor(_clearColor);
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                        1000.0 / Double(ImGui::GetIO().Framerate), Double(ImGui::GetIO().Framerate));
        }

        _transformation = Matrix4::translation(_cubePos) * _rotation;


        _shader.setLightPositions({{1.4f, 1.0f, 0.75f, 0.0f}})
                .setDiffuseColor(_color)
                .setAmbientColor(Color3::fromHsv({_color.hue(), 1.0f, 0.3f}))
                .setTransformationMatrix(_transformation)
                .setNormalMatrix(_transformation.normalMatrix())
                .setProjectionMatrix(_projection)
                .draw(_mesh);

        /* Set appropriate states. If you only draw ImGui, it is sufficient to
just enable blending and scissor test in the constructor. */
        GL::Renderer::enable(GL::Renderer::Feature::Blending);
        GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
        GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
        GL::Renderer::disable(GL::Renderer::Feature::DepthTest);


        _imgui.drawFrame();

        /* Reset state. Only needed if you want to draw something else with
   different state after. */
        GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
        GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
        GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
        GL::Renderer::disable(GL::Renderer::Feature::Blending);

        swapBuffers();
        redraw();
    }

    void BaseApplication::viewportEvent(ViewportEvent& event) {
        GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});

        _imgui.relayout(Vector2{event.windowSize()} / event.dpiScaling(),
                        event.windowSize(), event.framebufferSize());
    }

    void BaseApplication::keyPressEvent(KeyEvent& event) {
        if (_imgui.handleKeyPressEvent(event)) return;
    }

    void BaseApplication::keyReleaseEvent(KeyEvent& event) {
        if (_imgui.handleKeyReleaseEvent(event)) return;
    }

    void BaseApplication::mousePressEvent(MouseEvent& event) {
        if (_imgui.handleMousePressEvent(event)) return;
        if(event.button() != MouseEvent::Button::Left) return;

        event.setAccepted();
    }

    void BaseApplication::mouseReleaseEvent(MouseEvent& event) {
        if (_imgui.handleMouseReleaseEvent(event)) return;

        //_color = Color3::fromHsv({_color.hue() + 50.0_degf, 1.0f, 1.0f});

        event.setAccepted();
        //redraw();
    }

    void BaseApplication::mouseMoveEvent(MouseMoveEvent& event) {
        if (_imgui.handleMouseMoveEvent(event)) return;
        if(!(event.buttons() & MouseMoveEvent::Button::Left)) return;

        Vector2 delta = 3.0f*Vector2{event.relativePosition()}/Vector2{windowSize()};

        _rotation =
                Matrix4::rotationX(Rad{delta.y()})*
                _rotation*
                Matrix4::rotationY(Rad{delta.x()});

        event.setAccepted();
        redraw();
    }

    void BaseApplication::mouseScrollEvent(MouseScrollEvent& event) {
        if (_imgui.handleMouseScrollEvent(event)) {
        /* Prevent scrolling the page */
            event.setAccepted();
            return;
        }
    }

    void BaseApplication::textInputEvent(TextInputEvent& event) {
        if (_imgui.handleTextInputEvent(event)) return;
    }
}