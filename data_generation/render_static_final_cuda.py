import bpy
import bmesh
import sys
import os
import random
import math
import pickle as pkl
import numpy as np
import glob
import shutil
import re
import json
import torch

## add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utilities import (deselect_all, load_mesh, replace_environment_map,
                           render,
                           render_multilabel_mask,
                           replace_texture_for_object,
                           enable_backface_culling,
                           toggle_backface_transparency,
                           randomize_env_map_rotation,
                           configure_backface_culling
                           )
from src.cam_utils import (compute_lookat_point_and_set_camera, export_obj_builtin_keep_order,
                           export_worldspace_ply_builtin, randomize_camera_position_around_target,
                           get_intrinsic_K_from_camera,
                           get_camera_projection_matrix, save_blender_projection, save_camera_parameters,
                           save_object_transform, save_projection_stages,
                           randomize_fx_fy_from_camera, backup_camera_state,
                           restore_camera_state, dump_object_transform, save_camera_extrinsic_for_pytorch3d,
                           get_camera_extrinsic_for_pytorch3d_black_magic_1,
                           get_camera_extrinsic_for_pytorch3d_black_magic_2)
from pathlib import Path
from src.flame_utils import FLAME_Sampler
from src.utilities_clean import render_individual_binary_masks, render_multilabel_mask, render_normal_map_material, render_uv_map
from src.hair_utils import load_hair, sample_hair

DEBUG_PROJECTION_STAGES = False


# ══════════════════════════════════════════════════════════════════════════
#  Flexible hair sampling (works with any directory structure)
# ══════════════════════════════════════════════════════════════════════════

HAIR_EXTENSIONS = ('.obj', '.ply', '.fbx')

def sample_hair_flexible(hair_dir):
    """
    Find all hair files under any upsampled_hairstyle/ subdir of hair_dir and pick one at random.
    Excludes guiding/ and other low-resolution subfolders.
    """
    files = []
    for upsampled_dir in glob.glob(os.path.join(hair_dir, "**", "upsampled_hairstyle"), recursive=True):
        for root, _, fs in os.walk(upsampled_dir):
            for f in fs:
                if f.lower().endswith(HAIR_EXTENSIONS):
                    files.append(os.path.join(root, f))

    if not files:
        raise ValueError(f"No valid hair files ({HAIR_EXTENSIONS}) found under: {hair_dir}")

    chosen = random.choice(files)
    print(f"[Hair] Sampled: {chosen}  (from {len(files)} candidates)")
    return chosen


# ══════════════════════════════════════════════════════════════════════════
#  Gender-aware texture / hair helpers
# ══════════════════════════════════════════════════════════════════════════

def classify_textures_by_gender(texture_dir):
    """Split texture files into male / female lists based on filename keywords."""
    all_files = glob.glob(os.path.join(texture_dir, '*'))
    all_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    male_textures = []
    female_textures = []

    for f in all_files:
        basename = os.path.basename(f).lower()
        if 'female' in basename or 'woman' in basename:
            female_textures.append(f)
        elif 'male' in basename or 'man' in basename:
            male_textures.append(f)
        else:
            # Ambiguous — add to both pools so nothing is lost
            male_textures.append(f)
            female_textures.append(f)

    print(f"[Textures] {len(male_textures)} male, {len(female_textures)} female "
          f"(from {len(all_files)} total)")
    return male_textures, female_textures


def sample_gender(female_prob=0.5):
    """Return 'female' or 'male' based on probability."""
    return 'female' if random.random() < female_prob else 'male'


# ══════════════════════════════════════════════════════════════════════════

def clean_hair(curve_obj):
    bpy.data.objects.remove(curve_obj, do_unlink=True)


def seed_everything(seed=42):
    print(f"Seeding everything with seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image(
    png_out_file,
    engine='CYCLES',
    render_samples=64,
    res_percentage=100,
    background_visible=True
):
    configure_backface_culling(
        enable=False, scope="ALL", affect_viewport=True,
        affect_materials=True, adjust_nodes=True, recalc_normals=False, verbose=True
    )

    current_mode = bpy.context.object.mode
    bpy.ops.object.mode_set(mode='OBJECT')

    engine = engine.upper()
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.filepath = png_out_file
    scene.render.resolution_percentage = res_percentage
    scene.render.film_transparent = not background_visible

    if engine == 'CYCLES':
        scene.cycles.device = 'GPU'
        scene.cycles.samples = render_samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.use_denoising = False
        scene.render.use_compositing = True
        scene.render.use_sequencer = False
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'None'
        scene.display_settings.display_device = 'sRGB'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '8'
        scene.render.image_settings.file_format = 'PNG'
        scene.render.dither_intensity = 1.0
        scene.render.filter_size = 1.5
        if scene.world:
            scene.world.use_nodes = True
    elif engine == 'BLENDER_EEVEE':
        scene.eevee.taa_render_samples = render_samples
        scene.eevee.use_gtao = True
        scene.eevee.use_bloom = True
    else:
        raise ValueError("Unknown engine: use 'CYCLES' or 'EEVEE'")

    bpy.ops.render.render(write_still=True)
    bpy.ops.object.mode_set(mode=current_mode)
    toggle_backface_transparency(enable=True)


def disable_unwanted_modifiers():
    scene = bpy.context.scene
    GEOM_MODS = {
        'SUBSURF','TRIANGULATE','DECIMATE','MIRROR','ARRAY','BEVEL','SOLIDIFY','SKIN',
        'SCREW','WELD','BOOLEAN','REMESH','MULTIRES',
        'DISPLACE','ARMATURE','LATTICE','CURVE','SHRINKWRAP','SIMPLE_DEFORM','CAST',
        'HOOK','WARP','MESH_DEFORM','SURFACE_DEFORM','SMOOTH','CORRECTIVE_SMOOTH',
        'LAPLACIANSMOOTH','WAVE','NODES',
    }
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for m in obj.modifiers:
                if m.type in GEOM_MODS:
                    m.show_render = False
                    m.show_viewport = False
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.data.shape_keys:
            for kb in obj.data.shape_keys.key_blocks:
                kb.value = 0.0
    for obj in bpy.data.objects:
        for ps in getattr(obj, "particle_systems", []):
            if ps.settings:
                ps.settings.render_type = 'NONE'
                ps.settings.use_render_emitter = False
    if scene.render.engine == 'CYCLES':
        if hasattr(scene.cycles, "use_adaptive_subdivision"):
            scene.cycles.use_adaptive_subdivision = False
        for mat in bpy.data.materials:
            if hasattr(mat, "cycles"):
                mat.cycles.displacement_method = 'BUMP'
    scene.render.use_simplify = False


def rename_compositor_outputs(out_dir):
    rename_map = {
        r"hair\d+\.png": os.path.join("binary_masks", "hair_clean_binary_mask.png"),
        r"depth\d+\.png": "depth.png",
        r"depth_not_normalized\d+\.exr": "depth_not_normalized.exr"
    }
    for pattern, new_name in rename_map.items():
        for fname in os.listdir(out_dir):
            if re.fullmatch(pattern, fname):
                src_path = os.path.join(out_dir, fname)
                dst_path = os.path.join(out_dir, new_name)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.move(src_path, dst_path)
                print(f"Moved {fname} -> {dst_path}")


def render_single_image(
    mesh_file, hair_file, out_dir,
    flame_shape_db=False, flame_expression_db=False,
    resolution_percentage=100, texture_path=None, envmap_path=None,
    background_visible=False,
    randomize_left_right_angle=None, randomize_up_down_angle=None,
    randomize_camera_tilt=None, randomize_camera_lookat=None,
    randomize_camera_distance=None,
    randomize_lookat_offset_x=None, randomize_lookat_offset_y=None,
    randomize_lookat_offset_z=None,
    randomize_focal_length_range=None,
    randomize_env_map_rotation_x=None, randomize_env_map_rotation_y=None,
    randomize_env_map_rotation_z=None,
    render_engine='BLENDER_EEVEE', mask_type='binary',
    render_normal_map=True, render_uv_image=True, save_output_mesh=None,
    hair_color=0.1, hair_width=0.00007, render_samples=64,
    orig_head_path="./assets/head_prior.obj"
):
    if isinstance(mesh_file, (str, os.PathLike, list)):
        if isinstance(mesh_file, (str, os.PathLike)):
            mesh_ext = os.path.splitext(mesh_file)[1].lower()
            if mesh_ext == ".txt":
                with open(mesh_file, 'r') as f:
                    mesh_files = [line.strip() for line in f.readlines() if line.strip()]
            elif mesh_ext in [".obj", ".fbx", ".ply"]:
                mesh_files = [mesh_file]
            else:
                raise ValueError("mesh_file must be a mesh path, FLAME_Sampler, or list.")
        else:
            mesh_files = mesh_file
        if not mesh_files:
            raise ValueError(f"No valid mesh files found in {mesh_file}")
        mesh_file = random.choice(mesh_files)
        load_mesh(mesh_file, obj_name='flame_template')
    elif isinstance(mesh_file, FLAME_Sampler):
        vertices, flame_params = mesh_file.sample_flame_face(
            shape_from_db=flame_shape_db, expression_from_db=flame_expression_db,
            sample_eye_pose=True,
        )
        mesh = bpy.data.objects.get('flame_template')
        if mesh is not None:
            mesh.data.vertices.foreach_set("co", [co for v in vertices for co in v])
        else:
            raise RuntimeError("Object 'flame_template' not found.")
    else:
        raise ValueError("mesh_file must be a mesh path or FLAME_Sampler.")

    disable_unwanted_modifiers()

    obj = bpy.data.objects.get('flame_template')
    if obj is None:
        raise RuntimeError("Object 'flame_template' not found after loading mesh.")

    # Load hair
    hair_ext = os.path.splitext(hair_file)[1].lower()
    if hair_ext == ".txt":
        with open(hair_file, 'r') as f:
            hair_files = [line.strip() for line in f.readlines() if line.strip()]
        if not hair_files:
            raise ValueError(f"No valid hair files found in {hair_file}")
        hair_path = random.choice(hair_files)
    else:
        hair_path = sample_hair(hair_file)

    curve_obj, hair_color = load_hair(hair_path, apply_transform=True, head_mesh=obj,
                                       hair_color=hair_color, hair_width=hair_width,
                                       orig_head_path=orig_head_path)

    material_name = 'Material.001'
    mat = bpy.data.materials.get(material_name)
    if mat is None:
        raise RuntimeError(f"Material '{material_name}' not found.")
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    if texture_path is not None:
        replace_texture_for_object(object_name='flame_template', texture_node_name="face_texture",
                                   new_texture_path=texture_path)
    if envmap_path is not None:
        replace_environment_map(env_node_name="Environment Texture", new_env_map_path=envmap_path)

    if randomize_camera_lookat:
        compute_lookat_point_and_set_camera(obj_name='flame_template', camera_name='Camera')

    if randomize_camera_distance is not None or randomize_up_down_angle is not None or \
       randomize_left_right_angle is not None or randomize_lookat_offset_x is not None or \
       randomize_camera_tilt is not None:
        if randomize_lookat_offset_x is not None or randomize_lookat_offset_y is not None or \
           randomize_lookat_offset_z is not None:
            lookat_offset_pct = (
                random.uniform(randomize_lookat_offset_x[0], randomize_lookat_offset_x[1]),
                random.uniform(randomize_lookat_offset_y[0], randomize_lookat_offset_y[1]),
                random.uniform(randomize_lookat_offset_z[0], randomize_lookat_offset_z[1])
            )
        else:
            lookat_offset_pct = (0.0, 0.0, 0.0)
        delta_theta = random.uniform(math.radians(randomize_left_right_angle[0]),
                                     math.radians(randomize_left_right_angle[1])) if randomize_left_right_angle else 0.0
        delta_phi = random.uniform(math.radians(randomize_up_down_angle[0]),
                                   math.radians(randomize_up_down_angle[1])) if randomize_up_down_angle else 0.0
        delta_roll = random.uniform(math.radians(randomize_camera_tilt[0]),
                                    math.radians(randomize_camera_tilt[1])) if randomize_camera_tilt else 0.0
        randomize_camera_position_around_target(
            camera_name='Camera', target_object_name='flame_template',
            delta_theta=delta_theta, delta_phi=delta_phi, delta_roll=delta_roll,
            lookat_offset_pct=lookat_offset_pct, world_forward_up="-ZY",
        )

    if randomize_env_map_rotation_x is not None or randomize_env_map_rotation_y is not None or \
       randomize_env_map_rotation_z is not None:
        randomize_env_map_rotation(
            randomize_env_map_rotation_x=randomize_env_map_rotation_x,
            randomize_env_map_rotation_y=randomize_env_map_rotation_y,
            randomize_env_map_rotation_z=randomize_env_map_rotation_z,
        )

    if randomize_focal_length_range is not None:
        randomize_fx_fy_from_camera(camera_name='Camera')

    K = get_intrinsic_K_from_camera(camera_name='Camera')
    file_name = "output"
    png_out_file = os.path.join(out_dir, file_name + '.png')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    bpy.data.scenes['Scene'].node_tree.nodes['File Output'].base_path = out_dir
    bpy.data.scenes['Scene'].node_tree.nodes['File Output.001'].base_path = out_dir

    curve_obj.matrix_world = obj.matrix_world.copy()
    bpy.context.view_layer.update()

    export_obj_builtin_keep_order(
        object_name='flame_template',
        filepath=os.path.join(out_dir, file_name + '_mesh.obj'),
        apply_modifiers=False, triangulate=False, include_normals=False
    )

    render_image(png_out_file, engine=render_engine, res_percentage=resolution_percentage,
                 background_visible=background_visible, render_samples=render_samples)

    rename_compositor_outputs(out_dir)
    clean_hair(curve_obj)

    if save_output_mesh == 'ply':
        export_worldspace_ply_builtin(
            object_name='flame_template',
            filepath=os.path.join(out_dir, file_name + '_mesh_post_render.ply'),
            apply_modifiers=False, triangulate=False, use_render_levels=False, include_normals=False
        )
    if save_output_mesh == 'obj':
        export_obj_builtin_keep_order(
            object_name='flame_template',
            filepath=os.path.join(out_dir, file_name + '_mesh_post_render.obj'),
            apply_modifiers=False, triangulate=False, include_normals=False
        )

    save_camera_parameters(camera_name='Camera',
                           output_file=os.path.join(out_dir, file_name + '_camera_post_render.pkl'))
    save_blender_projection(camera_name='Camera',
                            file_path=os.path.join(out_dir, file_name + '_blender_projection_post_render.pkl'))
    save_camera_extrinsic_for_pytorch3d(
        bpy.data.objects.get("Camera"),
        filepath=os.path.join(out_dir, file_name + '_pytorch_camera_extrinsic_post_render.pkl'),
        func=get_camera_extrinsic_for_pytorch3d_black_magic_1
    )
    save_object_transform(object_name='flame_template',
                          file_path=os.path.join(out_dir, file_name + '_object_transform_post_render.pkl'))

    if render_uv_image:
        render_uv_map(renderer="EEVEE", output_path=os.path.join(out_dir, file_name + '_uv.png'))

    pkl_path = os.path.dirname(os.path.abspath(__file__)) + "/assets/FLAME_masks.pkl"

    if render_normal_map:
        render_normal_map_material(renderer="EEVEE", output_path=os.path.join(out_dir, file_name + '_normal.png'))

    if mask_type == 'rgb':
        render_multilabel_mask(renderer="EEVEE", object_name="flame_template", pkl_path=pkl_path,
                               output_path=os.path.join(out_dir, file_name + '_vertex_label_mask.png'))
    elif mask_type == 'binary':
        render_individual_binary_masks(renderer="EEVEE", object_name="flame_template", pkl_path=pkl_path,
                                       output_dir=os.path.join(out_dir, 'binary_masks'))
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}. Use 'rgb' or 'binary'")

    inputs = {
        'texture_path': texture_path, 'envmap_path': envmap_path,
        'hair_file': hair_path, 'hair_color': hair_color, 'hair_width': hair_width,
        'render_engine': render_engine, 'mask_type': mask_type,
    }
    if isinstance(mesh_file, FLAME_Sampler):
        inputs['flame_params'] = "sampled_flame"
        with open(os.path.join(out_dir, file_name + '_flame_params.pkl'), 'wb') as f:
            pkl.dump(flame_params, f)
    else:
        inputs['mesh_file'] = mesh_file

    with open(os.path.join(out_dir, file_name + '_inputs.json'), 'w') as f:
        json.dump(inputs, f, indent=4)


# ══════════════════════════════════════════════════════════════════════════
#  CLI helpers
# ══════════════════════════════════════════════════════════════════════════

def parse_csv_arg(argv, name):
    try:
        val = argv[argv.index(name) + 1]
        vals = [float(i) for i in val.split(',')]
        assert len(vals) == 2
        return vals
    except Exception:
        return None

def parse_int_arg(argv, name, default=None):
    try:
        return int(argv[argv.index(name) + 1])
    except Exception:
        return default

def parse_float_arg(argv, name, default=None):
    try:
        return float(argv[argv.index(name) + 1])
    except Exception:
        return default

def parse_str_arg(argv, name, default=None):
    try:
        return argv[argv.index(name) + 1]
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("Render engine:", bpy.context.scene.render.engine)

    prefs = bpy.context.preferences.addons['cycles'].preferences
    print("Available devices:")
    prefs.get_devices()
    for device in prefs.devices:
        print(f"  - {device.name} (type: {device.type}, use: {device.use})")

    argv = sys.argv
    mesh_file = argv[argv.index('--mesh_file') + 1]

    # ── FLAME setup ──
    if mesh_file in ['FLAME', 'FLAME2020', 'FLAME2023', 'FLAME2023_nojaw']:
        shape_scale = parse_int_arg(argv, '--shape_scale', 1.0)
        expression_scale = parse_int_arg(argv, '--expression_scale', 1.0)
        eye_yaw = parse_int_arg(argv, '--eye_yaw', 15)
        eye_pitch = parse_int_arg(argv, '--eye_pitch', 30)
        eye_roll = parse_int_arg(argv, '--eye_roll', 1)
        jaw_opening = parse_int_arg(argv, '--jaw_opening', 30)
        jaw_open_prob = parse_float_arg(argv, '--jaw_open_prob', 0.5)
        eyelid_prob = parse_float_arg(argv, '--eyelid_prob', 0.0)
        eyelid_range = parse_float_arg(argv, '--eyelid_range', 1.0)
        path_to_flame = parse_str_arg(argv, '--path_to_flame', str(Path(__file__).parent / "assets"))

        mesh_file = FLAME_Sampler(
            path_to_flame_model=path_to_flame, model_type=mesh_file,
            shape_scale=shape_scale, expression_scale=expression_scale,
            jaw_opening=jaw_opening, eye_yaw=eye_yaw, eye_pitch=eye_pitch,
            eye_roll=eye_roll, jaw_open_prob=jaw_open_prob,
            eyelid_prob=eyelid_prob, eyelid_range=eyelid_range,
            use_eyelids=eyelid_prob > 0.0,
        )

        # Move FLAME model to GPU for faster inference
        if torch.cuda.is_available():
            mesh_file.flame = mesh_file.flame.cuda()
            print("[GPU] FLAME model moved to CUDA")
    elif Path(mesh_file).is_file():
        mesh_ext = os.path.splitext(mesh_file)[1].lower()
        if mesh_ext == ".txt":
            with open(mesh_file, 'r') as f:
                mesh_file = [line.strip() for line in f.readlines() if line.strip()]
    else:
        raise ValueError("mesh_file must be a path to a mesh file or a FLAME model type.")

    # ── Gender-aware hair setup ──
    #    Use --hair_file_male / --hair_file_female for gendered hair directories.
    #    Falls back to --hair_file if gendered args are not provided.
    hair_file_male = parse_str_arg(argv, '--hair_file_male')
    hair_file_female = parse_str_arg(argv, '--hair_file_female')
    hair_file_fallback = parse_str_arg(argv, '--hair_file')  # legacy / single-gender

    hair_color = parse_float_arg(argv, '--hair_color', 0.1)
    hair_width = parse_float_arg(argv, '--hair_width', 0.00007)
    orig_head_path = parse_str_arg(argv, '--orig_head_path', './assets/head_prior.obj')

    # ── Gender probability ──
    female_prob = parse_float_arg(argv, '--female_prob', 0.5)

    # ── Output / general ──
    out_dir = argv[argv.index('--out_dir') + 1]
    render_samples = parse_int_arg(argv, '--render_samples', 64)
    resolution_percentage = parse_int_arg(argv, '--resolution_percentage', 100)
    background_visible = bool(parse_int_arg(argv, '--background_visible', 0))
    seed = parse_int_arg(argv, '--seed', 42)
    start_index = parse_int_arg(argv, '--start_index')
    end_index = parse_int_arg(argv, '--end_index')
    overwrite = bool(parse_int_arg(argv, '--overwrite', 0))

    render_engine = parse_str_arg(argv, '--render_engine', 'CYCLES').upper()

    # ── Texture ──
    texture_path = parse_str_arg(argv, '--texture_path')

    # Classify textures by gender if texture_path is a directory
    male_textures, female_textures = [], []
    if texture_path is not None and os.path.isdir(texture_path):
        male_textures, female_textures = classify_textures_by_gender(texture_path)

    # ── Envmap ──
    envmap_path = parse_str_arg(argv, '--envmap_path')

    # ── FLAME databases ──
    flame_shape_db = parse_str_arg(argv, '--flame_shape_db', False)
    if flame_shape_db:
        with open(flame_shape_db, 'r') as f:
            flame_shape_db = [line.strip() for line in f.readlines() if line.strip()]
        assert len(flame_shape_db) > 0
        print(f"Loaded {len(flame_shape_db)} shape ids")

    flame_expression_db = parse_str_arg(argv, '--flame_expression_db', False)
    if flame_expression_db:
        with open(flame_expression_db, 'r') as f:
            flame_expression_db = [line.strip() for line in f.readlines() if line.strip()]
        assert len(flame_expression_db) > 0
        print(f"Loaded {len(flame_expression_db)} expression vectors.")

    # ── Camera randomization ──
    randomize_left_right_angle = parse_csv_arg(argv, '--randomize_left_right_angle')
    randomize_up_down_angle = parse_csv_arg(argv, '--randomize_up_down_angle')
    randomize_camera_tilt = parse_csv_arg(argv, '--randomize_camera_tilt')
    randomize_camera_lookat = parse_int_arg(argv, '--randomize_camera_lookat')
    randomize_camera_distance = parse_csv_arg(argv, '--randomize_camera_distance')
    randomize_lookat_offset_x = parse_csv_arg(argv, '--randomize_lookat_offset_x')
    randomize_lookat_offset_y = parse_csv_arg(argv, '--randomize_lookat_offset_y')
    randomize_lookat_offset_z = parse_csv_arg(argv, '--randomize_lookat_offset_z')
    randomize_focal_length_range = parse_csv_arg(argv, '--randomize_focal_length_range')
    randomize_env_map_rotation_x = parse_csv_arg(argv, '--randomize_env_map_rotation_x')
    randomize_env_map_rotation_y = parse_csv_arg(argv, '--randomize_env_map_rotation_y')
    randomize_env_map_rotation_z = parse_csv_arg(argv, '--randomize_env_map_rotation_z')

    # ── Rendering modes ──
    num_images = parse_int_arg(argv, '--num_images')
    num_identities = parse_int_arg(argv, '--num_identities')
    num_lighting_setups = parse_int_arg(argv, '--num_lighting_setups')
    num_expressions = parse_int_arg(argv, '--num_expressions')
    views_per_expression = parse_int_arg(argv, '--views_per_expression')

    mask_type = parse_str_arg(argv, '--mask_type', 'binary').lower()
    save_output_mesh = parse_str_arg(argv, '--save_output_mesh', 'obj').lower()

    print("start_index =", start_index)
    print("end_index =", end_index)

    # Helper: resolve hair path for a given gender
    def get_hair_dir(gender):
        if gender == 'female' and hair_file_female:
            return hair_file_female
        elif gender == 'male' and hair_file_male:
            return hair_file_male
        elif hair_file_fallback:
            return hair_file_fallback
        else:
            raise ValueError(f"No hair path provided for gender '{gender}'. "
                             f"Use --hair_file_male / --hair_file_female or --hair_file.")

    # Helper: pick a texture for a given gender
    def get_texture_for_gender(gender):
        pool = female_textures if gender == 'female' else male_textures
        if pool:
            return random.choice(pool)
        return None

    # ══════════════════════════════════════════════════════════════════════
    #  MODE 1: Identity → Lighting → Expression → Camera  (WITH HAIR + GENDER)
    # ══════════════════════════════════════════════════════════════════════
    if num_identities is not None and num_lighting_setups is not None and \
       num_expressions is not None and views_per_expression is not None:

        print(f"Rendering {num_identities} identities x {num_lighting_setups} lighting x "
              f"{num_expressions} expressions x {views_per_expression} views  "
              f"(WITH HAIR, female_prob={female_prob})")

        cam_pose_backup, cam_intr_backup = backup_camera_state('Camera')

        identity_start = start_index if start_index is not None else 0
        identity_end = end_index if end_index is not None else num_identities

        # Pre-build genders for the full identity range.
        # For single-identity jobs, use Bernoulli sampling so female_prob works as expected.
        _n = identity_end - identity_start
        if _n == 1:
            _gender_list = ['female' if random.random() < female_prob else 'male']
        else:
            _n_female = round(_n * female_prob)
            _n_male = _n - _n_female
            _gender_list = ['female'] * _n_female + ['male'] * _n_male
            random.shuffle(_gender_list)

        _n_female = sum(1 for g in _gender_list if g == 'female')
        _n_male = _n - _n_female
        print(f"[Gender split] {_n_female} female, {_n_male} male out of {_n} identities")

        for identity_idx in range(identity_start, identity_end):
            identity_seed = seed + identity_idx * 100000
            seed_everything(identity_seed)

            print(f"\n{'='*60}")
            print(f"IDENTITY {identity_idx}")
            print(f"{'='*60}")

            # ── Assign gender for this identity (balanced across all identities) ──
            gender = _gender_list[identity_idx - identity_start]
            print(f"[Identity {identity_idx}] Gender: {gender}")

            # ── Sample shape once per identity ──
            if isinstance(mesh_file, FLAME_Sampler):
                shape_params, shape_file = mesh_file._sample_shape_params(shape_from_db=flame_shape_db)
                print(f"[Identity {identity_idx}] Sampled shape")

            # ── Sample texture once per identity (gender-matched) ──
            cached_texture_path = get_texture_for_gender(gender)
            if cached_texture_path:
                print(f"[Identity {identity_idx}] Texture: {os.path.basename(cached_texture_path)}")

            # ── Sample hair once per identity (gender-matched) ──
            hair_dir = get_hair_dir(gender)
            cached_hair_path = sample_hair_flexible(hair_dir)
            # If hair_color == -1 (random), pick a concrete value once per identity
            if hair_color < 0:
                cached_hair_color = random.random()  # [0, 1]
                print(f"[Identity {identity_idx}] Sampled random hair color: {cached_hair_color:.3f}")
            else:
                cached_hair_color = hair_color
            print(f"[Identity {identity_idx}] Hair: {os.path.basename(cached_hair_path)}")

            # ── Lighting loop ──
            for lighting_idx in range(num_lighting_setups):
                lighting_seed = identity_seed + lighting_idx * 10000
                seed_everything(lighting_seed)

                print(f"\n  {'='*50}")
                print(f"  LIGHTING SETUP {lighting_idx} (seed: {lighting_seed})")
                print(f"  {'='*50}")

                # Sample envmap once per lighting setup
                cached_envmap_path = None
                if envmap_path is not None:
                    if os.path.isdir(envmap_path):
                        envmap_files = glob.glob(os.path.join(envmap_path, '*'))
                        envmap_files = [f for f in envmap_files
                                        if f.lower().endswith(('.png','.jpg','.jpeg','.tiff','.hdr','.exr','.bmp'))]
                        if envmap_files:
                            cached_envmap_path = random.choice(envmap_files)
                            print(f"  [Lighting {lighting_idx}] Envmap: {os.path.basename(cached_envmap_path)}")
                    else:
                        cached_envmap_path = envmap_path

                # Sample envmap rotation once per lighting setup
                cached_env_rot_x, cached_env_rot_y, cached_env_rot_z = None, None, None
                if randomize_env_map_rotation_x is not None:
                    cached_env_rot_x = random.uniform(math.radians(randomize_env_map_rotation_x[0]),
                                                      math.radians(randomize_env_map_rotation_x[1]))
                if randomize_env_map_rotation_y is not None:
                    cached_env_rot_y = random.uniform(math.radians(randomize_env_map_rotation_y[0]),
                                                      math.radians(randomize_env_map_rotation_y[1]))
                if randomize_env_map_rotation_z is not None:
                    cached_env_rot_z = random.uniform(math.radians(randomize_env_map_rotation_z[0]),
                                                      math.radians(randomize_env_map_rotation_z[1]))

                # ── Expression loop ──
                for expression_idx in range(num_expressions):
                    expression_seed = lighting_seed + expression_idx * 100
                    seed_everything(expression_seed)

                    print(f"\n    Expression {expression_idx} (seed: {expression_seed})")

                    if isinstance(mesh_file, FLAME_Sampler):
                        expression_params, jaw_pose, expression_file, eyelid_params = \
                            mesh_file._sample_expression_params(expression_from_db=flame_expression_db)

                        eye_pose_deg = torch.zeros(1, 3)
                        eye_pose_deg[:, 0] = torch.rand(1) * (mesh_file.eye_yaw_range_deg[1] - mesh_file.eye_yaw_range_deg[0]) + mesh_file.eye_yaw_range_deg[0]
                        eye_pose_deg[:, 1] = torch.rand(1) * (mesh_file.eye_pitch_range_deg[1] - mesh_file.eye_pitch_range_deg[0]) + mesh_file.eye_pitch_range_deg[0]
                        eye_pose_deg[:, 2] = torch.rand(1) * (mesh_file.eye_roll_range_deg[1] - mesh_file.eye_roll_range_deg[0]) + mesh_file.eye_roll_range_deg[0]

                    # ── Camera view loop ──
                    for view_idx in range(views_per_expression):
                        view_seed = expression_seed + view_idx
                        seed_everything(view_seed)

                        out_dir_view = os.path.join(
                            out_dir,
                            f'identity_{identity_idx:06d}',
                            f'lighting_{lighting_idx:03d}',
                            f'expr_{expression_idx:03d}',
                            f'view_{view_idx:03d}'
                        )

                        if Path(out_dir_view).exists() and not overwrite:
                            print(f"    View {view_idx}: Skipping (exists)")
                            continue

                        print(f"    View {view_idx}: Rendering...")

                        # ── Generate FLAME vertices ──
                        if isinstance(mesh_file, FLAME_Sampler):
                            from src.FLAME.rotation_converter import batch_euler2axis, deg2rad
                            global_pose = torch.zeros(1, 3)
                            neck_pose = torch.zeros(1, 3)
                            eye_pose = batch_euler2axis(deg2rad(eye_pose_deg[:, :3]))
                            eye_pose_both_eyes = torch.cat([eye_pose, eye_pose], dim=1)
                            pose_params = torch.cat([global_pose, jaw_pose], dim=1)

                            # Move tensors to GPU if available
                            device = next(mesh_file.flame.parameters()).device
                            _sp = shape_params.to(device)
                            _ep = expression_params.to(device)
                            _pp = pose_params.to(device)
                            _np = neck_pose.to(device)
                            _eye = eye_pose_both_eyes.to(device)
                            _el = eyelid_params.to(device) if eyelid_params is not None else None

                            with torch.no_grad():
                                vertices, landmarks2d, landmarks3d, landmarks2d_mediapipe = mesh_file.flame(
                                    shape_params=_sp,
                                    expression_params=_ep,
                                    pose_params=_pp,
                                    neck_pose=_np,
                                    eye_pose_params=_eye,
                                    eyelid_params=_el
                                )

                            vertices_np = vertices.cpu().numpy().squeeze()
                            mesh_obj = bpy.data.objects.get('flame_template')
                            if mesh_obj is None:
                                raise RuntimeError("Object 'flame_template' not found.")
                            mesh_obj.data.vertices.foreach_set("co", [co for v in vertices_np for co in v])
                        else:
                            mesh_obj = bpy.data.objects.get('flame_template')

                        disable_unwanted_modifiers()

                        # ── Load hair (same for entire identity) ──
                        curve_obj, actual_hair_color = load_hair(
                            cached_hair_path, apply_transform=True, head_mesh=mesh_obj,
                            hair_color=cached_hair_color, hair_width=hair_width,
                            orig_head_path=orig_head_path
                        )

                        # ── Reassign material ──
                        material_name = 'Material.001'
                        mat = bpy.data.materials.get(material_name)
                        if mat is None:
                            raise RuntimeError(f"Material '{material_name}' not found.")
                        mesh_obj.data.materials.clear()
                        mesh_obj.data.materials.append(mat)

                        # ── Texture (cached per identity, gender-matched) ──
                        if cached_texture_path is not None:
                            replace_texture_for_object(
                                object_name='flame_template',
                                texture_node_name="face_texture",
                                new_texture_path=cached_texture_path
                            )

                        # ── Environment map (cached per lighting setup) ──
                        if cached_envmap_path is not None:
                            world = bpy.data.worlds['World']
                            if world.use_nodes:
                                env_node = world.node_tree.nodes.get("Environment Texture")
                                if env_node is not None:
                                    env_node.image = bpy.data.images.load(cached_envmap_path, check_existing=True)

                        # ── Environment map rotation (cached per lighting setup) ──
                        if cached_env_rot_x is not None or cached_env_rot_y is not None or cached_env_rot_z is not None:
                            world = bpy.data.worlds['World']
                            if world.use_nodes:
                                mapping_node = world.node_tree.nodes.get("Mapping")
                                if mapping_node is not None:
                                    default_value_x = 90.0
                                    mapping_node.inputs['Rotation'].default_value[0] = \
                                        cached_env_rot_x if cached_env_rot_x is not None else math.radians(default_value_x)
                                    mapping_node.inputs['Rotation'].default_value[1] = \
                                        cached_env_rot_y if cached_env_rot_y is not None else 0.0
                                    mapping_node.inputs['Rotation'].default_value[2] = \
                                        cached_env_rot_z if cached_env_rot_z is not None else 0.0

                        # ── Camera randomization (varies per view) ──
                        if randomize_camera_lookat:
                            compute_lookat_point_and_set_camera(obj_name='flame_template', camera_name='Camera')

                        if randomize_camera_distance is not None or randomize_up_down_angle is not None or \
                           randomize_left_right_angle is not None or randomize_lookat_offset_x is not None or \
                           randomize_camera_tilt is not None:
                            if randomize_lookat_offset_x is not None or randomize_lookat_offset_y is not None or \
                               randomize_lookat_offset_z is not None:
                                lookat_offset_pct = (
                                    random.uniform(randomize_lookat_offset_x[0], randomize_lookat_offset_x[1]),
                                    random.uniform(randomize_lookat_offset_y[0], randomize_lookat_offset_y[1]),
                                    random.uniform(randomize_lookat_offset_z[0], randomize_lookat_offset_z[1])
                                )
                            else:
                                lookat_offset_pct = (0.0, 0.0, 0.0)
                            delta_theta = random.uniform(math.radians(randomize_left_right_angle[0]),
                                                         math.radians(randomize_left_right_angle[1])) \
                                if randomize_left_right_angle else 0.0
                            delta_phi = random.uniform(math.radians(randomize_up_down_angle[0]),
                                                       math.radians(randomize_up_down_angle[1])) \
                                if randomize_up_down_angle else 0.0
                            delta_roll = random.uniform(math.radians(randomize_camera_tilt[0]),
                                                        math.radians(randomize_camera_tilt[1])) \
                                if randomize_camera_tilt else 0.0
                            randomize_camera_position_around_target(
                                camera_name='Camera', target_object_name='flame_template',
                                delta_theta=delta_theta, delta_phi=delta_phi, delta_roll=delta_roll,
                                lookat_offset_pct=lookat_offset_pct, world_forward_up="-ZY",
                            )

                        if randomize_focal_length_range is not None:
                            fx_new, fy_new = randomize_fx_fy_from_camera(camera_name='Camera')

                        K = get_intrinsic_K_from_camera(camera_name='Camera')

                        if not os.path.exists(out_dir_view):
                            os.makedirs(out_dir_view)

                        # ── Disable compositor file outputs (we only need output.png) ──
                        for node in bpy.data.scenes['Scene'].node_tree.nodes:
                            if node.type == 'OUTPUT_FILE':
                                node.mute = True

                        curve_obj.matrix_world = mesh_obj.matrix_world.copy()
                        bpy.context.view_layer.update()

                        file_name = "output"
                        export_obj_builtin_keep_order(
                            object_name='flame_template',
                            filepath=os.path.join(out_dir_view, file_name + '_mesh.obj'),
                            apply_modifiers=False, triangulate=False, include_normals=False
                        )

                        png_out_file = os.path.join(out_dir_view, file_name + '.png')
                        render_image(png_out_file, engine=render_engine, res_percentage=resolution_percentage,
                                     background_visible=background_visible, render_samples=render_samples)

                        rename_compositor_outputs(out_dir_view)
                        clean_hair(curve_obj)

                        if save_output_mesh == 'obj':
                            export_obj_builtin_keep_order(
                                object_name='flame_template',
                                filepath=os.path.join(out_dir_view, file_name + '_mesh_post_render.obj'),
                                apply_modifiers=False, triangulate=False, include_normals=False
                            )
                        elif save_output_mesh == 'ply':
                            export_worldspace_ply_builtin(
                                object_name='flame_template',
                                filepath=os.path.join(out_dir_view, file_name + '_mesh_post_render.ply'),
                                apply_modifiers=False, triangulate=False, use_render_levels=False, include_normals=False
                            )

                        save_camera_parameters(camera_name='Camera',
                                               output_file=os.path.join(out_dir_view, file_name + '_camera_post_render.pkl'))
                        save_blender_projection(camera_name='Camera',
                                                file_path=os.path.join(out_dir_view, file_name + '_blender_projection_post_render.pkl'))
                        save_camera_extrinsic_for_pytorch3d(
                            bpy.data.objects.get("Camera"),
                            filepath=os.path.join(out_dir_view, file_name + '_pytorch_camera_extrinsic_post_render.pkl'),
                            func=get_camera_extrinsic_for_pytorch3d_black_magic_1
                        )
                        save_object_transform(object_name='flame_template',
                                              file_path=os.path.join(out_dir_view, file_name + '_object_transform_post_render.pkl'))

                        # ── Save metadata (includes gender) ──
                        inputs = {
                            'identity_idx': identity_idx,
                            'gender': gender,
                            'lighting_idx': lighting_idx,
                            'expression_idx': expression_idx,
                            'view_idx': view_idx,
                            'identity_seed': identity_seed,
                            'lighting_seed': lighting_seed,
                            'expression_seed': expression_seed,
                            'view_seed': view_seed,
                            'texture_file': cached_texture_path,
                            'envmap_file': cached_envmap_path,
                            'hair_file': cached_hair_path,
                            'hair_color': actual_hair_color,
                            'hair_width': hair_width,
                            'texture_path': texture_path,
                            'envmap_path': envmap_path,
                            'render_engine': render_engine,
                        }
                        with open(os.path.join(out_dir_view, file_name + '_inputs.json'), 'w') as f:
                            json.dump(inputs, f, indent=4)

                        restore_camera_state('Camera', cam_pose_backup, cam_intr_backup)

    # ══════════════════════════════════════════════════════════════════════
    #  MODE 2: Single image
    # ══════════════════════════════════════════════════════════════════════
    elif num_images is None:
        seed_everything(seed)
        # For single image, sample gender and pick matching assets
        gender = sample_gender(female_prob)
        hair_dir = get_hair_dir(gender)
        single_texture = get_texture_for_gender(gender)

        render_single_image(
            mesh_file=mesh_file, hair_file=hair_dir,
            flame_shape_db=flame_shape_db, flame_expression_db=flame_expression_db,
            out_dir=out_dir, resolution_percentage=resolution_percentage,
            texture_path=single_texture, envmap_path=envmap_path,
            background_visible=background_visible,
            randomize_left_right_angle=randomize_left_right_angle,
            randomize_up_down_angle=randomize_up_down_angle,
            randomize_camera_tilt=randomize_camera_tilt,
            randomize_camera_lookat=randomize_camera_lookat,
            randomize_camera_distance=randomize_camera_distance,
            randomize_lookat_offset_x=randomize_lookat_offset_x,
            randomize_lookat_offset_y=randomize_lookat_offset_y,
            randomize_lookat_offset_z=randomize_lookat_offset_z,
            randomize_focal_length_range=randomize_focal_length_range,
            randomize_env_map_rotation_x=randomize_env_map_rotation_x,
            randomize_env_map_rotation_y=randomize_env_map_rotation_y,
            randomize_env_map_rotation_z=randomize_env_map_rotation_z,
            render_engine=render_engine, mask_type=mask_type,
            save_output_mesh=save_output_mesh,
            hair_color=hair_color, hair_width=hair_width,
            render_samples=render_samples, orig_head_path=orig_head_path,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  MODE 3: N independent images
    # ══════════════════════════════════════════════════════════════════════
    else:
        cam_pose_backup, cam_intr_backup = backup_camera_state('Camera')
        if start_index is not None and end_index is not None:
            image_rng = range(start_index, end_index + 1)
            assert end_index <= num_images
        else:
            image_rng = range(num_images)

        for i in image_rng:
            seed_everything(seed + i)

            # Sample gender per image
            gender = sample_gender(female_prob)
            hair_dir = get_hair_dir(gender)
            single_texture = get_texture_for_gender(gender)

            out_dir_ = os.path.join(out_dir, f'image_{i:06d}')
            if Path(out_dir_).exists() and not overwrite:
                print(f"Output directory {out_dir_} already exists. Skipping...")
                continue

            render_single_image(
                mesh_file=mesh_file, hair_file=hair_dir,
                flame_shape_db=flame_shape_db, flame_expression_db=flame_expression_db,
                out_dir=out_dir_, resolution_percentage=resolution_percentage,
                texture_path=single_texture, envmap_path=envmap_path,
                background_visible=background_visible,
                randomize_left_right_angle=randomize_left_right_angle,
                randomize_up_down_angle=randomize_up_down_angle,
                randomize_camera_tilt=randomize_camera_tilt,
                randomize_camera_lookat=randomize_camera_lookat,
                randomize_camera_distance=randomize_camera_distance,
                randomize_lookat_offset_x=randomize_lookat_offset_x,
                randomize_lookat_offset_y=randomize_lookat_offset_y,
                randomize_lookat_offset_z=randomize_lookat_offset_z,
                randomize_focal_length_range=randomize_focal_length_range,
                randomize_env_map_rotation_x=randomize_env_map_rotation_x,
                randomize_env_map_rotation_y=randomize_env_map_rotation_y,
                randomize_env_map_rotation_z=randomize_env_map_rotation_z,
                render_engine=render_engine, mask_type=mask_type,
                save_output_mesh=save_output_mesh,
                hair_color=hair_color, hair_width=hair_width,
                render_samples=render_samples, orig_head_path=orig_head_path,
            )
            restore_camera_state('Camera', cam_pose_backup, cam_intr_backup)


if __name__ == '__main__':
    main()
