import numpy as np
import dearpygui.dearpygui as dpg
import time
from rich import print

from raytracelib.orbit_camera import OrbitCamera


class GUI:

    def __init__(
        self,
        renderer,
        width=1920,
        height=1080,
        radius=1.4,
        fovy=50,
        mv_cameras=None,
        continuous_update=False,
        # is_using_mv_cameras_dims=False
    ):
        """
        TODO
        """
        print("\ninitializing GUI")

        # storing renderer reference
        self.renderer = renderer

        # mv cameras
        self.mv_cameras = mv_cameras
        if self.mv_cameras is not None:
            # override default width and height
            camera = self.mv_cameras[list(self.mv_cameras.data.keys())[0]][0]
            self.width, self.height = camera.width, camera.height
        else:
            self.width = width
            self.height = height

        # cameras
        self.orbit_camera = OrbitCamera(
            width=self.width, height=self.height, r=radius, fovy=fovy
        )

        self.active_camera = self.orbit_camera

        # store scene and camera
        print(f"resolution {self.width}x{self.height}")

        # self.is_using_mv_cameras_dims = is_using_mv_cameras_dims
        self.is_showing_mv_camera_gt = False

        # create render buffer
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)

        # utils
        self.continuous_update = continuous_update  # continuous rendering
        self.need_update = True  # when state changes, update frame

        dpg.create_context()
        self.register_dpg()
        self.draw()

    def __del__(self):
        dpg.destroy_context()

    def register_dpg(self):
        # register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.width,
                self.height,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        # register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.width, height=self.height):
            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=250, height=200):
            # button theme
            with dpg.theme():
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            with dpg.group(horizontal=True):
                dpg.add_text("rays generation time: ")
                dpg.add_text("no data", tag="_ray_gen_time")

            with dpg.group(horizontal=True):
                dpg.add_text("render time: ")
                dpg.add_text("no data", tag="_render_time")

            with dpg.group(horizontal=True):
                dpg.add_text("FPS: ")
                dpg.add_text("no data", tag="_frame_time")

            # rendering options
            with dpg.group():  # dpg.collapsing_header(label="Options", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.renderer.active_shaders = [app_data]
                    self.need_update = True
                    print(f"[INFO] rendering mode: {app_data}")

                dpg.add_combo(
                    self.renderer.renders_shaders,
                    label="shader",
                    default_value=self.renderer.active_shaders[0],
                    callback=callback_change_mode,
                )

                # which hit
                def callback_change_which_hit(sender, app_data):
                    if app_data == "first":
                        self.renderer.renders_options["render_second_hit"] = False
                    else:
                        self.renderer.renders_options["render_second_hit"] = True
                    self.need_update = True
                    print(f"[INFO] showing {app_data} hit")

                if "render_second_hit" in self.renderer.renders_options:
                    dpg.add_combo(
                        ("first", "second"),
                        label="which hit",
                        default_value="first",
                        callback=callback_change_which_hit,
                    )

                # which camera
                def callback_change_which_camera(sender, app_data):

                    if app_data == "orbit_camera":
                        self.active_camera = self.orbit_camera
                        # disable gt visualization
                        self.is_showing_mv_camera_gt = False
                        # TODO: disable error computation
                        # TODO: enable fov slider

                    else:
                        mode_idx = app_data.split(" ")[0]
                        split, cam_idx = mode_idx.split("_")
                        self.active_camera = self.mv_cameras[split][int(cam_idx)]
                        # TODO: disable fov slider
                        # if self.is_using_mv_cameras_dims:
                        # TODO:
                        # self.is_showing_mv_camera_gt = True
                        # pass

                        # at a later time, change the item's configuration
                        # dpg.configure_item("shader", items=["gt"] + self.renderer.renders_shaders)
                        # dpg.configure_item("vfov", enabled=False)
                        # activate error computation if rendering rgb
                        # allow gt visualization
                        # enable error shader

                    self.need_update = True
                    print(f"[INFO] camera {app_data} selected")

                # create list of available cameras
                available_cameras = ["orbit_camera"]
                if self.mv_cameras is not None:
                    if "test" in self.mv_cameras.data:
                        available_cameras += [
                            f"test_{i} ({camera.camera_idx})"
                            for i, camera in enumerate(self.mv_cameras["test"])
                        ]
                    if "train" in self.mv_cameras.data:
                        available_cameras += [
                            f"train_{i} ({camera.camera_idx})"
                            for i, camera in enumerate(self.mv_cameras["train"])
                        ]

                dpg.add_combo(
                    available_cameras,
                    label="which camera",
                    default_value=available_cameras[0],
                    callback=callback_change_which_camera,
                )

                # TODO: reactivate bg_color picker
                # # bg_color picker
                # def callback_change_bg(sender, app_data):
                #     self.renderer.bg_color = torch.tensor(
                #         app_data[:3], dtype=torch.float32, device="cuda"
                #     )
                #     self.need_update = True
                # dpg.add_color_edit(
                #     self.renderer.default_bg_color,
                #     label="bg color",
                #     width=200,
                #     tag="_color_editor",
                #     no_alpha=True,
                #     callback=callback_change_bg,
                # )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.orbit_camera.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(
                    label="vfov",
                    min_value=15,
                    max_value=120,
                    format="%d deg",
                    default_value=self.orbit_camera.fovy,
                    callback=callback_set_fovy,
                )

                # # mesh selection slider
                # def callback_set_mesh_id(sender, app_data):
                #     self.renderer.mesh_id = app_data
                #     self.need_update = True

                # dpg.add_slider_int(
                #     label="which mesh",
                #     min_value=0,
                #     max_value=len(self.renderer.tensor_meshes)-1,
                #     format="%d",
                #     default_value=self.renderer.mesh_id,
                #     callback=callback_set_mesh_id,
                # )

        # register camera handler

        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.orbit_camera.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.orbit_camera.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.orbit_camera.pan(dx, dy)
            self.need_update = True

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="viewer", width=self.width, height=self.height, resizable=False
        )

        # global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.setup_dearpygui()
        # dpg.show_metrics()
        dpg.show_viewport()

    def draw(self):
        # if new frame is needed
        if self.need_update:

            if self.is_showing_mv_camera_gt:
                # TODO: show gt (currenly broken)
                render = self.active_camera.get_rgb()
                # print("gt shape", render.shape)
            else:

                frame_time_start = time.time()

                # render
                renders = self.renderer.render(
                    self.active_camera, verbose=False, debug=False
                )

                # if renders is a dict
                # get first active shader in renderer.active_shaders
                if isinstance(renders, dict):
                    active_render_mode = self.renderer.active_renders_modes[0]
                    # print("keys", renders[active_render_mode].keys())
                    active_shader_key = self.renderer.active_shaders[0]
                    # TODO: reactivate which hit for specific methods
                    if "render_second_hit" in self.renderer.renders_options:
                        if self.renderer.renders_options["render_second_hit"]:
                            active_shader_key = "sh_" + active_shader_key
                    render = renders[active_render_mode][active_shader_key]
                    # TODO: if more than one render per shader is available, show selected
                    # print("render ndim", render.ndim)
                    # print("render shape", render.shape)
                    if render.ndim == 3:
                        render = render[:, 0, :]
                    # print("render shape", render.shape)
                    render = render.reshape(self.height, self.width, -1)
                    # print("render reshaped", render.shape)
                    # print(render.min(), render.max())
                    # scale to [0, 1] if needed
                    if np.max(render) > 1:
                        render = render / np.max(render)

                    # TODO: sometimes going segfault
                    # print(render.shape)

                    if render.shape[-1] == 1:
                        # repeat third channel 3 times
                        render = np.repeat(render, 3, axis=-1)
                else:
                    render = renders

                frame_time_end = time.time()

            # update gui render buffer
            print("render.shape", render.shape)
            self.render_buffer = render
            # print("render_buffer shape", self.render_buffer.shape)

            # collect stats from renderer profiler
            stats = {}
            if self.renderer.profiler is not None:

                ray_gen_time = self.renderer.profiler.get_last_time("ray_gen") * 1000
                # avg_ray_gen_time = profiler.get_avg_time("ray_gen") * 1000
                stats["ray_gen_time"] = f"{ray_gen_time:.4f}ms"

                render_time = self.renderer.profiler.get_last_time("render") * 1000
                # avg_render_time = profiler.get_avg_time("render") * 1000
                stats["render_time"] = f"{render_time:.4f}ms"

                frame_time = frame_time_end - frame_time_start
                # avg_frame_time = avg_ray_gen_time + avg_render_time
                stats["frame_time"] = f"{int(1/frame_time)} FPS"

            # if continuos update is disabled, set need_update to False
            if not self.continuous_update:
                self.need_update = False

            # update gui texture
            dpg.set_value("_texture", self.render_buffer)

            # TODO: reactivate
            # update gui stats
            for k, v in stats.items():
                dpg.set_value(f"_{k}", v)

    def render(self):
        while dpg.is_dearpygui_running():
            self.draw()
            dpg.render_dearpygui_frame()
