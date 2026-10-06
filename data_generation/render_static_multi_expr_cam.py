import bpy
import bmesh
import sys
import os
import random
import math
import pickle as pkl
# import numpy
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
from src.cam_utils import compute_lookat_point_and_set_camera, export_obj_builtin_keep_order, export_worldspace_ply_builtin, randomize_camera_position_around_target, \
                          get_intrinsic_K_from_camera, \
                          get_camera_projection_matrix, save_blender_projection, save_camera_parameters, \
                          save_object_transform, save_projection_stages, \
                          randomize_fx_fy_from_camera, backup_camera_state, \
                          restore_camera_state, dump_object_transform, save_camera_extrinsic_for_pytorch3d, \
                          get_camera_extrinsic_for_pytorch3d_black_magic_1, get_camera_extrinsic_for_pytorch3d_black_magic_2
from pathlib import Path
from src.flame_utils import FLAME_Sampler

from src.utilities_clean import render_individual_binary_masks, render_multilabel_mask, render_normal_map_material, render_uv_map

# DEBUG_PROJECTION_STAGES = True
DEBUG_PROJECTION_STAGES = False


import random
import numpy as np
import torch

def seed_everything(seed=42):
    print(f"Seeding everything with seed: {seed}")
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)


def render(
    png_out_file,
    engine='CYCLES',
    render_samples=64,
    res_percentage=100,
    background_visible=True
):
    configure_backface_culling(
        enable=False,          # turn OFF any culling-like behavior
        scope="ALL",
        affect_viewport=True,
        affect_materials=True,
        adjust_nodes=True,     # removes Geometry.Backfacing → Mix/Transparent links
        recalc_normals=False,
        verbose=True
    )

    # enable_backface_culling(enable=False)  # Disable backface culling for all materials
    # toggle_backface_transparency(enable=True)  # Disable backface transparency for all materials
    # toggle_backface_transparency(enable=False)  # Disable backface transparency for all materials
    
    # Store current object mode
    current_mode = bpy.context.object.mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Set render engine
    engine = engine.upper()
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.filepath = png_out_file
    scene.render.resolution_percentage = res_percentage

    # Set background visibility
    scene.render.film_transparent = not background_visible

    # Engine-specific settings with comprehensive reset
    if engine == 'CYCLES':
        # CRITICAL: Ensure CYCLES settings are completely reset
        # This fixes white image issue when switching back from EEVEE
        scene.cycles.device = 'GPU'
        scene.cycles.samples = render_samples
        scene.cycles.use_adaptive_sampling = True
        
        # Reset any compositor settings that might interfere
        scene.render.use_compositing = True
        scene.render.use_sequencer = False
        
        # Reset color management to standard values
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'None'
        scene.display_settings.display_device = 'sRGB'
        
        # Reset render settings that might be modified by EEVEE functions
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '8'
        scene.render.image_settings.file_format = 'PNG'
        scene.render.dither_intensity = 1.0
        scene.render.filter_size = 1.5
        
        # Ensure world uses nodes for proper HDRI lighting
        if scene.world:
            scene.world.use_nodes = True
        
        print(f"[DEBUG] CYCLES render setup complete - engine: {scene.render.engine}")
        
    elif engine == 'BLENDER_EEVEE':
        scene.eevee.taa_render_samples = render_samples
        scene.eevee.use_gtao = True
        scene.eevee.use_bloom = True
    else:
        raise ValueError("Unknown engine: use 'CYCLES' or 'EEVEE'")

    # Render the image
    bpy.ops.render.render(write_still=True)

    # Restore previous object mode
    bpy.ops.object.mode_set(mode=current_mode)
    
    # enable_backface_culling(enable=True)  # Re-enable backface culling for all materials
    toggle_backface_transparency(enable=True)  # Re-enable backface transparency for all materials


def disable_unwanted_modifiers():

    scene = bpy.context.scene

    # Turn off all geometry-changing modifiers (viewport + render)
    GEOM_MODS = {
        'SUBSURF','TRIANGULATE','DECIMATE','MIRROR','ARRAY','BEVEL','SOLIDIFY','SKIN',
        'SCREW','WELD','BOOLEAN','REMESH','MULTIRES',         # topology changers
        'DISPLACE','ARMATURE','LATTICE','CURVE','SHRINKWRAP','SIMPLE_DEFORM','CAST',
        'HOOK','WARP','MESH_DEFORM','SURFACE_DEFORM','SMOOTH','CORRECTIVE_SMOOTH',
        'LAPLACIANSMOOTH','WAVE','NODES',                     # deformers / GeoNodes
    }

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for m in obj.modifiers:
                if m.type in GEOM_MODS:
                    m.show_render = False
                    m.show_viewport = False

    # Disable shape keys
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.data.shape_keys:
            for kb in obj.data.shape_keys.key_blocks:
                kb.value = 0.0

    # Disable particles/hair that could add geo
    for obj in bpy.data.objects:
        for ps in getattr(obj, "particle_systems", []):
            if ps.settings:
                ps.settings.render_type = 'NONE'
                ps.settings.use_render_emitter = False

    # Neutralize true displacement / adaptive subdiv (Cycles)
    if scene.render.engine == 'CYCLES':
        if hasattr(scene.cycles, "use_adaptive_subdivision"):
            scene.cycles.use_adaptive_subdivision = False
        for mat in bpy.data.materials:
            if hasattr(mat, "cycles"):
                mat.cycles.displacement_method = 'BUMP'  # no real displacement

    scene.render.use_simplify = False  # avoid hidden simplifications



def render_single_image(
    mesh_file,
    out_dir,
    flame_shape_db=False,
    flame_expression_db=False,
    resolution_percentage=100,
    texture_path=None,
    envmap_path=None,
    background_visible=False,
    randomize_left_right_angle=None,
    randomize_up_down_angle=None,
    randomize_camera_tilt=None,
    randomize_camera_lookat=None,
    randomize_camera_distance=None,
    randomize_lookat_offset_x=None,
    randomize_lookat_offset_y=None,
    randomize_lookat_offset_z=None,
    randomize_focal_length_range=None,
    randomize_env_map_rotation_x=None,
    randomize_env_map_rotation_y=None,
    randomize_env_map_rotation_z=None,
    render_engine='BLENDER_EEVEE',
    mask_type='binary',
    render_normal_map=True,
    render_uv_image=True,
    save_output_mesh=None,
): 
    
    if isinstance(mesh_file, (str, os.PathLike, list)):

        if isinstance(mesh_file, (str, os.PathLike)):
            print("mesh fiiiiileeeeee", mesh_file)
            mesh_ext = os.path.splitext(mesh_file)[1].lower()
            
            if mesh_ext == ".txt": 
                ## read the list of mesh paths from the text file into a list
                with open(mesh_file, 'r') as f:
                    mesh_file = f.readlines()
                mesh_files = [line.strip() for line in mesh_file if line.strip()]
            elif mesh_ext in [".obj", ".fbx", ".ply"]:
                mesh_files = [mesh_file]
            else: 
                raise ValueError("mesh_file must be a path to a mesh file or a FLAME_Sampler instance or a list of mesh files.")
        else:
            mesh_files = mesh_file
            
        if len(mesh_files) == 0:
            raise ValueError(f"No valid mesh files found in {mesh_file}")
        
        ## randomly select one mesh file from the list
        mesh_file = random.choice(mesh_files)
        print (f"Selecting mesh file: {mesh_file}")
        
        load_mesh(mesh_file, obj_name='flame_template') 

    elif isinstance(mesh_file, FLAME_Sampler):
        flame_sampler = mesh_file
        vertices, flame_params = flame_sampler.sample_flame_face(
            shape_from_db=flame_shape_db,
            expression_from_db=flame_expression_db,
            sample_eye_pose=True,  
        ) 
        print(f"Sampled vertices: {vertices.shape}")

        # replace the vertices in the mesh from the np.array with vertices 
        mesh = bpy.data.objects.get('flame_template')
        if mesh is not None:
            mesh.data.vertices.foreach_set("co", [co for v in vertices for co in v])
        else:
            raise RuntimeError("Object 'flame_template' not found.")

    else: 
        raise ValueError("mesh_file must be a path to a mesh file or a FLAME_Sampler instance.")
    disable_unwanted_modifiers()

    ## reassign material 
    obj = bpy.data.objects.get('flame_template')
    if obj is None:
        obj = bpy.data.objects.get('flame_template')
        if obj is None:
            raise RuntimeError("Object 'flame_template' not found.")

        print(f"Materials assigned to '{obj.name}':")
        for slot in obj.material_slots:
            print("  -", slot.material.name)

        raise RuntimeError("Target object 'flame_template' not found after loading mesh.")

    # Reassign consistent material
    material_name = 'Material.001'
    mat = bpy.data.materials.get(material_name)
    if mat is None:
        mat = obj.active_material
        if mat and mat.use_nodes:
            print(f"Nodes in material '{mat.name}':")
            for node in mat.node_tree.nodes:
                print(f"  - {node.name} ({node.type})")
        raise RuntimeError(f"Material '{material_name}' not found in blend file.")

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    ''' replace the texture ''' 
    if texture_path is not None:
        replace_texture_for_object(
            object_name='flame_template',          # <-- name of the object
            texture_node_name="face_texture",   # <-- node label in the Shader Editor
            new_texture_path=texture_path        # <-- full path to new texture
        )
    if envmap_path is not None:
        replace_environment_map(
            env_node_name="Environment Texture",        # <-- node name visible in your screenshot
            new_env_map_path=envmap_path  # <-- full path to new HDRI file
        )
        
    if randomize_camera_lookat:
        compute_lookat_point_and_set_camera(
            obj_name='flame_template',  # <-- name of the object to look at
            camera_name='Camera'        # <-- name of the camera
        )

    if randomize_camera_distance is not None or randomize_up_down_angle is not None or randomize_left_right_angle is not None \
        or randomize_lookat_offset_x is not None or randomize_camera_tilt is not None:
        print('randomize_camera_distance', randomize_camera_distance)
        print('randomize_up_down_angle', randomize_up_down_angle)
        print('randomize_left_right_angle', randomize_left_right_angle)
        print('randomize_lookat_offset_x', randomize_lookat_offset_x)
        print('randomize_lookat_offset_y', randomize_lookat_offset_y)
        print('randomize_lookat_offset_z', randomize_lookat_offset_z)
        # randomize_camera_position_around_target(
        #     camera_name='Camera',  # <-- name of the camera
        #     target_name='flame_template',  # <-- name of the object to look at
        #     distance_multipliers=randomize_camera_distance,  # <-- distance from the object
        #     # randomize_angle=randomize_camera_angle,  # <-- randomize camera angle
        # )
        
        if randomize_lookat_offset_x is not None or randomize_lookat_offset_y is not None or randomize_lookat_offset_z is not None:
            lookat_offset_pct = (
                random.uniform(randomize_lookat_offset_x[0], randomize_lookat_offset_x[1]),
                random.uniform(randomize_lookat_offset_y[0], randomize_lookat_offset_y[1]),
                random.uniform(randomize_lookat_offset_z[0], randomize_lookat_offset_z[1])
            )
        else: 
            # set to zeros 
            lookat_offset_pct = (0.0, 0.0, 0.0)
        print('lookat_offset_pct', lookat_offset_pct)

        if randomize_left_right_angle is not None:        
            delta_theta = random.uniform(math.radians(randomize_left_right_angle[0]), 
                                         math.radians(randomize_left_right_angle[1]))
        else: 
            delta_theta = 0.0 
            
        if randomize_up_down_angle is not None:
            delta_phi = random.uniform(math.radians(randomize_up_down_angle[0]), 
                                       math.radians(randomize_up_down_angle[1]))
        else:
            delta_phi = 0.0
            
        if randomize_camera_tilt is not None:
            delta_roll = random.uniform(math.radians(randomize_camera_tilt[0]), 
                                        math.radians(randomize_camera_tilt[1]))
        else:
            delta_roll = 0.0
        
        
        randomize_camera_position_around_target(
            camera_name='Camera',
            target_object_name='flame_template',
            # distance_multipliers=(0.8, 1.2),
            # delta_theta=random.uniform(math.radians(-120), math.radians(120)),
            delta_theta=delta_theta,
            # delta_phi=random.uniform(math.radians(-65), math.radians(65))
            delta_phi=delta_phi,
            delta_roll=delta_roll,  # Pass roll directly to the function
            # lookat_offset_pct=(0.10, 0.10, 0.10),  # <-- offset from the object center
            lookat_offset_pct=lookat_offset_pct,  # <-- offset from the object center
            world_forward_up="-ZY", # if FLAME is left untransformed, top of the head is pointing at +Y, preferred
            # world_forward_up="YZ", # if FLAME is rotated such that the top of the head is UP (axis +Z) 
        )

    if randomize_env_map_rotation_x is not None or randomize_env_map_rotation_y is not None or randomize_env_map_rotation_z is not None:
        env_rot_x, env_rot_y, env_rot_z = randomize_env_map_rotation(
            randomize_env_map_rotation_x=randomize_env_map_rotation_x,
            randomize_env_map_rotation_y=randomize_env_map_rotation_y,
            randomize_env_map_rotation_z=randomize_env_map_rotation_z,
        )

        print(f"Rotation the env map by: X={env_rot_x:.2f}°, Y={env_rot_y:.2f}°, Z={env_rot_z:.2f}°")

    else: 
        print("No randomization applied to the environment map rotation.")


    if randomize_focal_length_range is not None:
        fx_new, fy_new = randomize_fx_fy_from_camera(camera_name='Camera')


    K = get_intrinsic_K_from_camera(
        camera_name='Camera', 
    )
    # cam_params = get_camera_projection_matrix(camera_name='Camera')  
    

    # print('Camera intrinsic matrix K:\n', K)

    # file_name = os.path.split(mesh_file)[1]
    # file_name = os.path.splitext(file_name)[0]
    file_name = "output"
    png_out_file = os.path.join(out_dir, file_name+'.png')
    # render(png_out_file, res_percentage=100) 
    
    # create output directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # # camera_out_file = os.path.join(out_dir, file_name+'_camera.pkl')

    # # export_worldspace_ply_builtin(
    # #     object_name='flame_template',
    # #     filepath=os.path.join(out_dir, file_name+'_premesh.ply'),
    # #     # apply_modifiers=True,
    # #     apply_modifiers=False,
    # #     # triangulate=True,
    # #     triangulate=False,
    # #     # use_render_levels=True,
    # #     use_render_levels=False,
    # #     include_normals=False
    # # )

    # # export_worldspace_ply_builtin(
    # #     object_name='flame_template',
    # #     filepath=os.path.join(out_dir, file_name+'_premesh_mod.ply'),
    # #     apply_modifiers=True,
    # #     # apply_modifiers=False,
    # #     # triangulate=True,
    # #     triangulate=False,
    # #     # use_render_levels=True,
    # #     use_render_levels=False,
    # #     include_normals=False
    # # ) 

    # ### APPLYING MODIFIERS CHANGES THE CAMERA (even through the function above that should have nothing to do with the camera)
    
    # # print("SAVING CAMERA PARAMETERS")
    # # save_camera_parameters(
    # #     camera_name='Camera',
    # #     output_file=camera_out_file,
    # # )

    # # print("SAVING BLENDER PROJECTION")
    # # save_blender_projection(
    # #     camera_name='Camera',
    # #     file_path=os.path.join(out_dir, file_name+'_blender_projection.pkl')
    # # )

    # # ## for debug only
    # # print("SAVING PROJECTION STAGES no mod")
    # # save_projection_stages(
    # #     object_name="flame_template",
    # #     mode="vertices",
    # #     camera=bpy.data.objects.get("Camera"),
    # #     apply_modifiers=False,
    # #     filepath=os.path.join(out_dir, file_name+'_flame_template_projection_no_mod.pkl')
    # # )

    # print("SAVING PROJECTION STAGES")
    # save_projection_stages(
    #     object_name="flame_template",
    #     mode="vertices",
    #     camera=bpy.data.objects.get("Camera"),
    #     filepath=os.path.join(out_dir, file_name+'_flame_template_projection.pkl')
    # )

    # ## for debug only
    # object_transform_out_file = os.path.join(out_dir, file_name+'_object_transform.pkl')
    # save_object_transform('flame_template', object_transform_out_file)


    # # export_worldspace_ply_builtin(
    # #     object_name='flame_template',
    # #     filepath=os.path.join(out_dir, file_name+'_mesh.ply'),
    # #     # apply_modifiers=True,
    # #     apply_modifiers=False,
    # #     # triangulate=True,
    # #     triangulate=False,
    # #     # use_render_levels=True,
    # #     use_render_levels=False,
    # #     include_normals=False
    # # )

    export_obj_builtin_keep_order(
        object_name='flame_template',
        filepath=os.path.join(out_dir, file_name+'_mesh.obj'),
        # apply_modifiers=True,
        apply_modifiers=False,
        # triangulate=True,
        triangulate=False,
        # use_render_levels=True,
        # use_render_levels=False,
        include_normals=False
    )



    # export_obj_builtin_keep_order(
    #     object_name='flame_template',
    #     filepath=os.path.join(out_dir, file_name+'_mesh_mod.obj'),
    #     apply_modifiers=True,
    #     # apply_modifiers=False,
    #     # triangulate=True,
    #     triangulate=False,
    #     # use_render_levels=True,
    #     # use_render_levels=False,
    #     include_normals=False
    # )

    #     # if save_camera_parameters:
    # if True:
    #     camera_out_file = os.path.join(out_dir, file_name+'_camera_pre_render.pkl')

    #     save_camera_parameters(
    #         camera_name='Camera',
    #         output_file=camera_out_file,
    #     )

    #     # # print("SAVING BLENDER PROJECTION")
    #     # save_blender_projection(
    #     #     camera_name='Camera',
    #     #     file_path=os.path.join(out_dir, file_name+'_blender_projection_pre_render.pkl')
    #     # )

    #     # save_camera_extrinsic_for_pytorch3d(
    #     #     bpy.data.objects.get("Camera"),
    #     #     filepath=os.path.join(out_dir, file_name+'_pytorch_camera_extrinsic_pre_render.pkl')
    #     # )

    ### SOME WORDS OF ADVICE 
    ## 1) RENDER STUFF FIRST, EXPORT STUFF SECOND (for some reason, the scene graph evaluation thinks that happens might affect the positions of cameras and objects slightly)
    render(png_out_file, engine=render_engine, res_percentage=resolution_percentage, background_visible=background_visible) 
    
    # bpy.ops.wm.save_as_mainfile(filepath="test.blend", copy=True)


    if save_output_mesh == 'ply':
        export_worldspace_ply_builtin(
            object_name='flame_template',
            filepath=os.path.join(out_dir, file_name+'_mesh_post_render.ply'),
            # apply_modifiers=True,
            apply_modifiers=False,
            # triangulate=True,
            triangulate=False,
            # use_render_levels=True,
            use_render_levels=False,
            include_normals=False
        )
        
        # export_worldspace_ply_builtin(
        #     object_name='flame_template',
        #     filepath=os.path.join(out_dir, file_name+'_mesh_post_render_mod.ply'),
        #     # apply_modifiers=True,
        #     apply_modifiers=True,
        #     # triangulate=True,
        #     triangulate=False,
        #     # use_render_levels=True,
        #     use_render_levels=False,
        #     include_normals=False
        # )

        

    if save_output_mesh == 'obj':
        export_obj_builtin_keep_order(
            object_name='flame_template',
            filepath=os.path.join(out_dir, file_name+'_mesh_post_render.obj'),
            # apply_modifiers=True,
            apply_modifiers=False,
            # triangulate=True,
            triangulate=False,
            # use_render_levels=True,
            # use_render_levels=False,
            include_normals=False
        )


        # export_obj_builtin_keep_order(
        #     object_name='flame_template',
        #     filepath=os.path.join(out_dir, file_name+'_mesh_post_render_mod.obj'),
        #     # apply_modifiers=True,
        #     apply_modifiers=True,
        #     # triangulate=True,
        #     triangulate=False,
        #     # use_render_levels=True,
        #     # use_render_levels=False,
        #     include_normals=False
        # )

    if DEBUG_PROJECTION_STAGES:
        save_projection_stages(
            object_name="flame_template",
            mode="vertices",
            camera=bpy.data.objects.get("Camera"),
            filepath=os.path.join(out_dir, file_name+'_flame_template_projection_post_render.pkl')
        )


        save_projection_stages(
            object_name="flame_template",
            mode="vertices",
            camera=bpy.data.objects.get("Camera"),
            apply_modifiers=True,
            filepath=os.path.join(out_dir, file_name+'_flame_template_projection_post_render_mod.pkl')
        )

    
    
    # if save_camera_parameters:
    if True:
        camera_out_file = os.path.join(out_dir, file_name+'_camera_post_render.pkl')
        save_camera_parameters(
            camera_name='Camera',
            output_file=camera_out_file,
        )

        # print("SAVING BLENDER PROJECTION")
        save_blender_projection(
            camera_name='Camera',
            file_path=os.path.join(out_dir, file_name+'_blender_projection_post_render.pkl')
        )

        save_camera_extrinsic_for_pytorch3d(
            bpy.data.objects.get("Camera"),
            filepath=os.path.join(out_dir, file_name+'_pytorch_camera_extrinsic_post_render.pkl'), 
            func=get_camera_extrinsic_for_pytorch3d_black_magic_1
        )

        # save_camera_extrinsic_for_pytorch3d(
        #     bpy.data.objects.get("Camera"),
        #     filepath=os.path.join(out_dir, file_name+'_pytorch_camera_extrinsic_post_render_2.pkl'),
        #     func=get_camera_extrinsic_for_pytorch3d_black_magic_2
        # )

        object_transform_out_file = os.path.join(out_dir, file_name+'_object_transform_post_render.pkl')
        save_object_transform(
            object_name='flame_template',
            file_path=object_transform_out_file,
        )

    if render_uv_image: 
        render_uv_map(renderer="EEVEE", output_path=os.path.join(out_dir, file_name+'_uv.png'))

    # === MASK RENDERING OPTIONS ===
    pkl_path = os.path.dirname(os.path.abspath(__file__)) + "/assets/FLAME_masks.pkl"
    
    if render_normal_map:
        # Using new material-based approach to avoid white image issues
        render_normal_map_material(
            renderer="EEVEE",
            # object_name="flame_template",
            output_path= os.path.join(out_dir, file_name+'_normal.png')
        )

    if mask_type == 'rgb':
        # Render RGB multi-label mask (original approach)
        render_multilabel_mask(
            renderer="EEVEE",
            object_name="flame_template",
            pkl_path=pkl_path,
            output_path= os.path.join(out_dir, file_name + '_vertex_label_mask.png')
        )
        print(f"[INFO] RGB segmentation mask saved to: {os.path.join(out_dir, file_name + '_vertex_label_mask.png')}")
    
    elif mask_type == 'binary':
        # Render clean individual binary masks (new optimized approach)
        binary_masks_dir = os.path.join(out_dir, 'binary_masks')
        render_individual_binary_masks(
            renderer="EEVEE",
            object_name="flame_template",
            pkl_path=pkl_path,
            output_dir=binary_masks_dir,
            # render_engine=render_engine
        )
        print(f"[INFO] Clean binary masks saved to directory: {binary_masks_dir}")
    
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}. Use 'rgb' or 'binary'")
    
    ## create a dictionary with all the inputs 
    inputs = {
        'texture_path': texture_path,
        'envmap_path': envmap_path,
        'resolution_percentage': resolution_percentage,
        'background_visible': background_visible,
        'randomize_left_right_angle': randomize_left_right_angle,
        'randomize_up_down_angle': randomize_up_down_angle,
        'randomize_camera_tilt': randomize_camera_tilt,
        'randomize_camera_lookat': randomize_camera_lookat,
        'randomize_camera_distance': randomize_camera_distance,
        'randomize_lookat_offset_x': randomize_lookat_offset_x,
        'randomize_lookat_offset_y': randomize_lookat_offset_y,
        'randomize_lookat_offset_z': randomize_lookat_offset_z,
        'randomize_focal_length_range': randomize_focal_length_range,
        'render_engine': render_engine,
        'mask_type': mask_type,
        'delta_theta': delta_theta if randomize_left_right_angle is not None else 0.0,
        'delta_phi': delta_phi if randomize_up_down_angle is not None else 0.0,
        'delta_roll': delta_roll if randomize_camera_tilt is not None else 0.0,
        'lookat_offset_x': lookat_offset_pct[0] if randomize_lookat_offset_x is not None else 0.0,
        'lookat_offset_y': lookat_offset_pct[1] if randomize_lookat_offset_y is not None else 0.0,
        'lookat_offset_z': lookat_offset_pct[2] if randomize_lookat_offset_z is not None else 0.0,
        'fx_new': fx_new if randomize_focal_length_range is not None else K[0][0],
        'fy_new': fy_new if randomize_focal_length_range is not None else K[1][1],
        # 'envmap_rotation': (env_rot_x, env_rot_y, env_rot_z) if randomize_env_map_rotation is not None else (0.0, 0.0, 0.0)
    }

    if isinstance(mesh_file, FLAME_Sampler):
        inputs['flame_params'] = f"sampled_flame"
        # pickle the sampled flame 
        flame_params_file = os.path.join(out_dir, file_name + '_flame_params.pkl')
        with open(flame_params_file, 'wb') as f:
            pkl.dump(flame_params, f)
    else: 
        inputs['mesh_file'] = mesh_file

    # Save inputs to a JSON file
    import json
    inputs_file = os.path.join(out_dir, file_name + '_inputs.json')
    with open(inputs_file, 'w') as f:
        json.dump(inputs, f, indent=4)
    

    # ## dump camera matrices with pickle 
    # inputs_file = os.path.join(out_dir, file_name + '_cam.pkl')
    # with open(inputs_file, 'wb') as f:
    #     pkl.dump(cam_params, f)

    # bpy.ops.wm.save_as_mainfile(filepath = os.path.dirname(os.path.abspath(__file__)) + "/your_debug_scene.blend")



def main():
    import bpy
    print("Render engine:", bpy.context.scene.render.engine)

    # Test GPU render device
    prefs = bpy.context.preferences.addons['cycles'].preferences
    print("Available devices:")
    prefs.get_devices()
    for device in prefs.devices:
        print(f"  - {device.name} (type: {device.type}, use: {device.use})")

    # sys.exit(0)

    argv = sys.argv
    mesh_file = argv[argv.index('--mesh_file') + 1]
    

    if mesh_file in ['FLAME', 'FLAME2020', 'FLAME2023', 'FLAME2023_nojaw']: 
        try:
            shape_scale = int(argv[argv.index('--shape_scale') + 1])
        except Exception:
            shape_scale = 1.0
        try:
            expression_scale = int(argv[argv.index('--expression_scale') + 1])
        except Exception:
            expression_scale = 1.0 

        try: 
            eye_yaw = int(argv[argv.index('--eye_yaw') + 1])
        except Exception:
            eye_yaw = 15
        try:
            eye_pitch = int(argv[argv.index('--eye_pitch') + 1])
        except Exception:
            eye_pitch = 30
        try:
            eye_roll = int(argv[argv.index('--eye_roll') + 1])
        except Exception:
            eye_roll = 1
        try: 
            jaw_opening = int(argv[argv.index('--jaw_opening') + 1])
        except Exception:
            jaw_opening = 30

        try: 
            jaw_open_prob = float(argv[argv.index('--jaw_open_prob') + 1])
        except Exception:
            jaw_open_prob = 0.5

        try: 
            eyelid_prob = float(argv[argv.index('--eyelid_prob') + 1])
            print("----------------------------------------------------------------")
            print("eyelid_prob", eyelid_prob)
        except Exception:
            print("sth went wrong")
            eyelid_prob = 0.0

        try: 
            eyelid_range = float(argv[argv.index('--eyelid_range') + 1])
            print("----------------------------------------------------------------")
            print("eyelid_range", eyelid_range)
        except Exception:
            print("sth went wrong")
            eyelid_range = 1.0
            
        try:
            path_to_flame = argv[argv.index('--path_to_flame') + 1]
        except Exception: 
            path_to_flame = Path(__file__).parent / "assets"
        
       
        mesh_file = FLAME_Sampler(
            path_to_flame_model=path_to_flame,
            model_type=mesh_file,
            shape_scale=shape_scale,
            expression_scale=expression_scale,
            jaw_opening=jaw_opening,
            eye_yaw=eye_yaw,
            eye_pitch=eye_pitch,
            eye_roll=eye_roll,
            jaw_open_prob=jaw_open_prob,
            eyelid_prob=eyelid_prob,
            eyelid_range=eyelid_range, 
            use_eyelids=eyelid_prob > 0.0,
        )
    elif Path(mesh_file).is_file(): 
        mesh_ext = os.path.splitext(mesh_file)[1].lower()
        
        if mesh_ext == ".txt": 
            ## read the list of mesh paths from the text file into a list
            with open(mesh_file, 'r') as f:
                mesh_file = f.readlines()
            mesh_files = [line.strip() for line in mesh_file if line.strip()]
            mesh_file = mesh_files

    else: 
        raise ValueError("mesh_file must be a path to a mesh file or a FLAME_Sampler instance.")

    out_dir = argv[argv.index('--out_dir') + 1]

    try: 
        flame_shape_db = argv[argv.index('--flame_shape_db') + 1]
    except Exception:
        flame_shape_db = False

    if flame_shape_db is not False: 
        flame_shape_db_ext = os.path.splitext(flame_shape_db)[1].lower()
        assert flame_shape_db_ext == ".txt", f"Expected .txt file, got {flame_shape_db_ext}"
        ## read the list of shape ids from the text file into a list
        with open(flame_shape_db, 'r') as f:
            flame_shape_db = f.readlines()
        flame_shape_db = [line.strip() for line in flame_shape_db if line.strip()]
        assert len(flame_shape_db) > 0, f"No valid shape ids found in {flame_shape_db}"
        print(f"Loaded {len(flame_shape_db)} shape ids")

    try: 
        flame_expression_db = argv[argv.index('--flame_expression_db') + 1]
    except Exception:
        flame_expression_db = False 
        
    if flame_expression_db is not False:
        flame_expression_db_ext = os.path.splitext(flame_expression_db)[1].lower()
        assert flame_expression_db_ext == ".txt", f"Expected .txt file, got {flame_expression_db_ext}"
        ## read the list of expression ids from the text file into a list
        with open(flame_expression_db, 'r') as f:
            flame_expression_db = f.readlines()
        flame_expression_db = [line.strip() for line in flame_expression_db if line.strip()]
        assert len(flame_expression_db) > 0, f"No valid expression ids found in {flame_expression_db}"
        print(f"Loaded {len(flame_expression_db)} expression vectors.")

    try:
        resolution_percentage = int(argv[argv.index('--resolution_percentage') + 1])
    except Exception: 
        resolution_percentage = 100
        
    try:
        texture_path = argv[argv.index('--texture_path') + 1]
    except Exception: 
        texture_path = None
        
    try:
        envmap_path = argv[argv.index('--envmap_path') + 1]
    except Exception: 
        envmap_path = None
    
    try: 
        background_visible = int(argv[argv.index('--background_visible') + 1])
    except Exception:
        background_visible = False
        
    try: 
        randomize_left_right_angle = argv[argv.index('--randomize_left_right_angle') + 1]
        randomize_left_right_angle = [float(i) for i in randomize_left_right_angle.split(',')]
        assert len(randomize_left_right_angle) == 2, f"Expected 2 values, got {len(randomize_left_right_angle)}"
    except Exception:
        randomize_left_right_angle = None
        
            
    try: 
        randomize_up_down_angle = argv[argv.index('--randomize_up_down_angle') + 1]
        randomize_up_down_angle = [float(i) for i in randomize_up_down_angle.split(',')]
        assert len(randomize_up_down_angle) == 2, f"Expected 2 values, got {len(randomize_up_down_angle)}"
    except Exception:
        randomize_up_down_angle = None
        
    try: 
        randomize_camera_tilt = argv[argv.index('--randomize_camera_tilt') + 1]
        randomize_camera_tilt = [float(i) for i in randomize_camera_tilt.split(',')]
        assert len(randomize_camera_tilt) == 2, f"Expected 2 values, got {len(randomize_camera_tilt)}"
    except Exception:
        randomize_camera_tilt = None
        
        
    try: 
        randomize_camera_lookat = int(argv[argv.index('--randomize_camera_lookat') + 1])
    except Exception:
        randomize_camera_lookat = None
        
    try: 
        randomize_camera_distance = argv[argv.index('--randomize_camera_distance') + 1] 
        randomize_camera_distance = [float(i) for i in randomize_camera_distance.split(',')]
    except Exception:
        randomize_camera_distance = None
        
        
    try: 
        randomize_lookat_offset_x = argv[argv.index('--randomize_lookat_offset_x') + 1]
        randomize_lookat_offset_x = [float(i) for i in randomize_lookat_offset_x.split(',')]
        assert len(randomize_lookat_offset_x) == 2, f"Expected 2 values, got {len(randomize_lookat_offset_x)}"
    except Exception:
        randomize_lookat_offset_x = None
        
    try:
        randomize_lookat_offset_y = argv[argv.index('--randomize_lookat_offset_y') + 1]
        randomize_lookat_offset_y = [float(i) for i in randomize_lookat_offset_y.split(',')]
        assert len(randomize_lookat_offset_y) == 2, f"Expected 2 values, got {len(randomize_lookat_offset_y)}"
    except Exception:
        randomize_lookat_offset_y = None
    
    try:
        randomize_lookat_offset_z = argv[argv.index('--randomize_lookat_offset_z') + 1]
        randomize_lookat_offset_z = [float(i) for i in randomize_lookat_offset_z.split(',')]
        assert len(randomize_lookat_offset_z) == 2, f"Expected 2 values, got {len(randomize_lookat_offset_z)}"
    except Exception:
        randomize_lookat_offset_z = None
    
    try:
        randomize_focal_length_range = argv[argv.index('--randomize_focal_length_range') + 1]
        randomize_focal_length_range = [float(i) for i in randomize_focal_length_range.split(',')]
        assert len(randomize_focal_length_range) == 2, f"Expected 2 values, got {len(randomize_focal_length_range)}"
    except Exception:
        randomize_focal_length_range = None
        

    try: 
        randomize_env_map_rotation_x = argv[argv.index('--randomize_env_map_rotation_x') + 1]
        randomize_env_map_rotation_x = [float(i) for i in randomize_env_map_rotation_x.split(',')]
        assert len(randomize_env_map_rotation_x) == 2, f"Expected 2 values, got {len(randomize_env_map_rotation_x)}"
    except Exception:
        randomize_env_map_rotation_x = None
    

    try:
        randomize_env_map_rotation_y = argv[argv.index('--randomize_env_map_rotation_y') + 1]
        randomize_env_map_rotation_y = [float(i) for i in randomize_env_map_rotation_y.split(',')]
        assert len(randomize_env_map_rotation_y) == 2, f"Expected 2 values, got {len(randomize_env_map_rotation_y)}"
    except Exception:
        randomize_env_map_rotation_y = None

    try:
        randomize_env_map_rotation_z = argv[argv.index('--randomize_env_map_rotation_z') + 1]
        randomize_env_map_rotation_z = [float(i) for i in randomize_env_map_rotation_z.split(',')]
        assert len(randomize_env_map_rotation_z) == 2, f"Expected 2 values, got {len(randomize_env_map_rotation_z)}"
    except Exception:
        randomize_env_map_rotation_z = None

    try:
        render_engine = argv[argv.index('--render_engine') + 1].upper()
    except Exception:
        render_engine = 'CYCLES'  # default
        # render_engine = 'BLENDER_EEVEE'  
        
    try:
        num_images = int(argv[argv.index('--num_images') + 1])
    except Exception:
        num_images = None

    try:
        num_identities = int(argv[argv.index('--num_identities') + 1])
    except Exception:
        num_identities = None

    try:
        num_lighting_setups = int(argv[argv.index('--num_lighting_setups') + 1])
    except Exception:
        num_lighting_setups = None

    try:
        num_expressions = int(argv[argv.index('--num_expressions') + 1])
    except Exception:
        num_expressions = None

    try:
        views_per_expression = int(argv[argv.index('--views_per_expression') + 1])
    except Exception:
        views_per_expression = None

    try:
        mask_type = argv[argv.index('--mask_type') + 1].lower()
        assert mask_type in ['rgb', 'binary'], f"Invalid mask_type: {mask_type}. Use 'rgb' or 'binary'"
    except Exception:
        mask_type = 'binary'  # Default to binary masks
        

    try:
        save_output_mesh = argv[argv.index('--save_output_mesh') + 1].lower()
        assert save_output_mesh in ['obj', 'ply'], f"Invalid save_output_mesh: {save_output_mesh}. Use 'obj' or 'ply'"
    except Exception:
        save_output_mesh = 'obj'  # Default to not saving the output mesh


    try: 
        seed = int(argv[argv.index('--seed') + 1])
    except Exception:
        seed = 42

    try: 
        start_index = int(argv[argv.index('--start_index') + 1])
    except Exception:
        start_index = None

    try:
        end_index = int(argv[argv.index('--end_index') + 1])
    except Exception:
        end_index = None

    try:
        render_samples = int(argv[argv.index('--render_samples') + 1])
    except Exception:
        render_samples = 64  # Default value for render samples

    try: 
        overwrite = int(argv[argv.index('--overwrite') + 1])
    except Exception:
        overwrite = 0
    overwrite = bool(overwrite)

    print("start_index =", start_index)
    print("end_index =", end_index)

    # Check if we're using lighting-expression-camera hierarchy (identity -> lighting -> expressions -> camera views)
    if num_identities is not None and num_lighting_setups is not None and num_expressions is not None and views_per_expression is not None:
        print(f"Rendering {num_identities} identities x {num_lighting_setups} lighting setups x {num_expressions} expressions x {views_per_expression} views")
        print("Mode: Identity -> Lighting -> Expression -> Camera hierarchy (NO HAIR)")

        ## back up the camera state
        cam_pose_backup, cam_intr_backup = backup_camera_state('Camera')

        identity_start = start_index if start_index is not None else 0
        identity_end = end_index if end_index is not None else num_identities

        for identity_idx in range(identity_start, identity_end):
            # Seed for this identity (fixes shape and texture)
            identity_seed = seed + identity_idx * 100000
            seed_everything(identity_seed)

            print(f"\n{'='*60}")
            print(f"IDENTITY {identity_idx}")
            print(f"{'='*60}")

            # Sample shape once for this identity (if using FLAME)
            if isinstance(mesh_file, FLAME_Sampler):
                shape_params, shape_file = mesh_file._sample_shape_params(shape_from_db=flame_shape_db)
                print(f"[Identity {identity_idx}] Sampled shape")

            # Sample and cache texture for this identity
            cached_texture_path = None
            if texture_path is not None:
                import glob
                texture_files = glob.glob(os.path.join(texture_path, '*'))
                texture_files = [f for f in texture_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if texture_files:
                    cached_texture_path = random.choice(texture_files)
                    print(f"[Identity {identity_idx}] Sampled texture: {os.path.basename(cached_texture_path)}")

            # Loop through lighting setups
            for lighting_idx in range(num_lighting_setups):
                # Seed for this lighting setup (fixes environment map and lighting rotation)
                lighting_seed = identity_seed + lighting_idx * 10000
                seed_everything(lighting_seed)

                print(f"\n  {'='*50}")
                print(f"  LIGHTING SETUP {lighting_idx} (seed: {lighting_seed})")
                print(f"  {'='*50}")

                # Sample and cache environment map for this lighting setup (fixed for all expressions in this setup)
                cached_envmap_path = None
                if envmap_path is not None:
                    import glob
                    if os.path.isdir(envmap_path):
                        envmap_files = glob.glob(os.path.join(envmap_path, '*'))
                        envmap_files = [f for f in envmap_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.hdr', '.exr', '.bmp'))]
                        if envmap_files:
                            cached_envmap_path = random.choice(envmap_files)
                            print(f"  [Lighting {lighting_idx}] Sampled environment map: {os.path.basename(cached_envmap_path)}")
                    else:
                        cached_envmap_path = envmap_path

                # Sample and cache environment map rotation for this lighting setup (fixed for all expressions)
                cached_env_rot_x = None
                cached_env_rot_y = None
                cached_env_rot_z = None
                if randomize_env_map_rotation_x is not None or randomize_env_map_rotation_y is not None or randomize_env_map_rotation_z is not None:
                    if randomize_env_map_rotation_x is not None:
                        cached_env_rot_x = random.uniform(math.radians(randomize_env_map_rotation_x[0]),
                                                         math.radians(randomize_env_map_rotation_x[1]))
                    if randomize_env_map_rotation_y is not None:
                        cached_env_rot_y = random.uniform(math.radians(randomize_env_map_rotation_y[0]),
                                                         math.radians(randomize_env_map_rotation_y[1]))
                    if randomize_env_map_rotation_z is not None:
                        cached_env_rot_z = random.uniform(math.radians(randomize_env_map_rotation_z[0]),
                                                         math.radians(randomize_env_map_rotation_z[1]))
                    print(f"  [Lighting {lighting_idx}] Environment map rotation: x={cached_env_rot_x}, y={cached_env_rot_y}, z={cached_env_rot_z}")

                # Loop through expressions (all with same lighting)
                for expression_idx in range(num_expressions):
                    # Seed for this expression (fixes expression parameters only)
                    expression_seed = lighting_seed + expression_idx * 100
                    seed_everything(expression_seed)

                    print(f"\n    Expression {expression_idx} (seed: {expression_seed})")

                    # Sample expression once for this expression group
                    if isinstance(mesh_file, FLAME_Sampler):
                        expression_params, jaw_pose, expression_file, eyelid_params = mesh_file._sample_expression_params(expression_from_db=flame_expression_db)

                        # Sample eye pose once for this expression
                        eye_pose_deg = torch.zeros(1, 3)
                        eye_pose_deg[:, 0] = torch.rand(1) * (mesh_file.eye_yaw_range_deg[1] - mesh_file.eye_yaw_range_deg[0]) + mesh_file.eye_yaw_range_deg[0]
                        eye_pose_deg[:, 1] = torch.rand(1) * (mesh_file.eye_pitch_range_deg[1] - mesh_file.eye_pitch_range_deg[0]) + mesh_file.eye_pitch_range_deg[0]
                        eye_pose_deg[:, 2] = torch.rand(1) * (mesh_file.eye_roll_range_deg[1] - mesh_file.eye_roll_range_deg[0]) + mesh_file.eye_roll_range_deg[0]

                    # Loop through camera views for this expression
                    for view_idx in range(views_per_expression):
                        # Seed for this camera view (varies camera only)
                        view_seed = expression_seed + view_idx
                        seed_everything(view_seed)

                        out_dir_view = os.path.join(out_dir, f'identity_{identity_idx:06d}', f'lighting_{lighting_idx:03d}', f'expr_{expression_idx:03d}', f'view_{view_idx:03d}')

                        if Path(out_dir_view).exists() and not overwrite:
                            print(f"    View {view_idx}: Skipping (exists)")
                            continue

                        print(f"    View {view_idx}: Rendering...")

                        # For FLAME, generate vertices with fixed shape and fixed expression
                        if isinstance(mesh_file, FLAME_Sampler):
                            global_pose = torch.zeros(1, 3)
                            neck_pose = torch.zeros(1, 3)

                            from src.FLAME.rotation_converter import batch_euler2axis, deg2rad
                            eye_pose = batch_euler2axis(deg2rad(eye_pose_deg[:,:3]))
                            eye_pose_both_eyes = torch.cat([eye_pose, eye_pose], dim=1)
                            pose_params = torch.cat([global_pose, jaw_pose], dim=1)

                            with torch.no_grad():
                                vertices, landmarks2d, landmarks3d, landmarks2d_mediapipe = mesh_file.flame(
                                    shape_params=shape_params,  # Fixed shape for identity
                                    expression_params=expression_params,  # Fixed expression for this group
                                    pose_params=pose_params,
                                    neck_pose=neck_pose,
                                    eye_pose_params=eye_pose_both_eyes,
                                    eyelid_params=eyelid_params
                                )

                            vertices_np = vertices.cpu().numpy().squeeze()

                            # Load mesh template and update vertices
                            mesh_obj = bpy.data.objects.get('flame_template')
                            if mesh_obj is not None:
                                mesh_obj.data.vertices.foreach_set("co", [co for v in vertices_np for co in v])
                            else:
                                raise RuntimeError("Object 'flame_template' not found.")

                            from src.utilities import replace_texture_for_object, replace_environment_map
                            disable_unwanted_modifiers()

                            # Reassign material
                            material_name = 'Material.001'
                            mat = bpy.data.materials.get(material_name)
                            if mat is None:
                                raise RuntimeError(f"Material '{material_name}' not found in blend file.")
                            mesh_obj.data.materials.clear()
                            mesh_obj.data.materials.append(mat)

                            # Replace texture (use cached texture path for this identity)
                            if cached_texture_path is not None:
                                replace_texture_for_object(
                                    object_name='flame_template',
                                    texture_node_name="face_texture",
                                    new_texture_path=cached_texture_path
                                )

                            # Replace environment map (use cached path for this lighting setup)
                            if cached_envmap_path is not None:
                                # Directly set the environment map without random sampling
                                world = bpy.data.worlds['World']
                                if world.use_nodes:
                                    env_node = world.node_tree.nodes.get("Environment Texture")
                                    if env_node is not None:
                                        env_node.image = bpy.data.images.load(cached_envmap_path, check_existing=True)

                            # Randomize camera (ONLY thing that varies per view)
                            if randomize_camera_lookat:
                                compute_lookat_point_and_set_camera(
                                    obj_name='flame_template',
                                    camera_name='Camera'
                                )

                            if randomize_camera_distance is not None or randomize_up_down_angle is not None or randomize_left_right_angle is not None \
                                or randomize_lookat_offset_x is not None or randomize_camera_tilt is not None:

                                if randomize_lookat_offset_x is not None or randomize_lookat_offset_y is not None or randomize_lookat_offset_z is not None:
                                    lookat_offset_pct = (
                                        random.uniform(randomize_lookat_offset_x[0], randomize_lookat_offset_x[1]),
                                        random.uniform(randomize_lookat_offset_y[0], randomize_lookat_offset_y[1]),
                                        random.uniform(randomize_lookat_offset_z[0], randomize_lookat_offset_z[1])
                                    )
                                else:
                                    lookat_offset_pct = (0.0, 0.0, 0.0)

                                if randomize_left_right_angle is not None:
                                    delta_theta = random.uniform(math.radians(randomize_left_right_angle[0]),
                                                                 math.radians(randomize_left_right_angle[1]))
                                else:
                                    delta_theta = 0.0

                                if randomize_up_down_angle is not None:
                                    delta_phi = random.uniform(math.radians(randomize_up_down_angle[0]),
                                                               math.radians(randomize_up_down_angle[1]))
                                else:
                                    delta_phi = 0.0

                                if randomize_camera_tilt is not None:
                                    delta_roll = random.uniform(math.radians(randomize_camera_tilt[0]),
                                                                math.radians(randomize_camera_tilt[1]))
                                else:
                                    delta_roll = 0.0

                                randomize_camera_position_around_target(
                                    camera_name='Camera',
                                    target_object_name='flame_template',
                                    delta_theta=delta_theta,
                                    delta_phi=delta_phi,
                                    delta_roll=delta_roll,
                                    lookat_offset_pct=lookat_offset_pct,
                                    world_forward_up="-ZY",
                                )

                            # Apply cached environment map rotation (fixed for this lighting setup)
                            if cached_env_rot_x is not None or cached_env_rot_y is not None or cached_env_rot_z is not None:
                                world = bpy.data.worlds['World']
                                if world.use_nodes:
                                    mapping_node = world.node_tree.nodes.get("Mapping")
                                    if mapping_node is not None:
                                        # Apply cached rotation values (with FLAME default offsets)
                                        default_value_x = 90.0  # FLAME uses 90° rotation
                                        mapping_node.inputs['Rotation'].default_value[0] = cached_env_rot_x if cached_env_rot_x is not None else math.radians(default_value_x)
                                        mapping_node.inputs['Rotation'].default_value[1] = cached_env_rot_y if cached_env_rot_y is not None else 0.0
                                        mapping_node.inputs['Rotation'].default_value[2] = cached_env_rot_z if cached_env_rot_z is not None else 0.0

                            if randomize_focal_length_range is not None:
                                fx_new, fy_new = randomize_fx_fy_from_camera(camera_name='Camera')

                            K = get_intrinsic_K_from_camera(camera_name='Camera')

                            # Create output directory
                            if not os.path.exists(out_dir_view):
                                os.makedirs(out_dir_view)

                            # Export mesh before render
                            file_name = "output"
                            export_obj_builtin_keep_order(
                                object_name='flame_template',
                                filepath=os.path.join(out_dir_view, file_name+'_mesh.obj'),
                                apply_modifiers=False,
                                triangulate=False,
                                include_normals=False
                            )

                            # Render
                            png_out_file = os.path.join(out_dir_view, file_name+'.png')
                            render(png_out_file, engine=render_engine, res_percentage=resolution_percentage,
                                   background_visible=background_visible, render_samples=render_samples)

                            # Export post-render mesh if requested
                            if save_output_mesh == 'obj':
                                export_obj_builtin_keep_order(
                                    object_name='flame_template',
                                    filepath=os.path.join(out_dir_view, file_name+'_mesh_post_render.obj'),
                                    apply_modifiers=False,
                                    triangulate=False,
                                    include_normals=False
                                )
                            elif save_output_mesh == 'ply':
                                export_worldspace_ply_builtin(
                                    object_name='flame_template',
                                    filepath=os.path.join(out_dir_view, file_name+'_mesh_post_render.ply'),
                                    apply_modifiers=False,
                                    triangulate=False,
                                    use_render_levels=False,
                                    include_normals=False
                                )

                            # Save camera parameters
                            camera_out_file = os.path.join(out_dir_view, file_name+'_camera_post_render.pkl')
                            save_camera_parameters(camera_name='Camera', output_file=camera_out_file)
                            save_blender_projection(
                                camera_name='Camera',
                                file_path=os.path.join(out_dir_view, file_name+'_blender_projection_post_render.pkl')
                            )
                            save_camera_extrinsic_for_pytorch3d(
                                bpy.data.objects.get("Camera"),
                                filepath=os.path.join(out_dir_view, file_name+'_pytorch_camera_extrinsic_post_render.pkl'),
                                func=get_camera_extrinsic_for_pytorch3d_black_magic_1
                            )
                            object_transform_out_file = os.path.join(out_dir_view, file_name+'_object_transform_post_render.pkl')
                            save_object_transform(object_name='flame_template', file_path=object_transform_out_file)

                            # Render UV and masks
                            render_uv_map(renderer="EEVEE", output_path=os.path.join(out_dir_view, file_name+'_uv.png'))

                            pkl_path = os.path.dirname(os.path.abspath(__file__)) + "/assets/FLAME_masks.pkl"
                            render_normal_map_material(renderer="EEVEE", output_path=os.path.join(out_dir_view, file_name+'_normal.png'))

                            if mask_type == 'binary':
                                binary_masks_dir = os.path.join(out_dir_view, 'binary_masks')
                                render_individual_binary_masks(
                                    renderer="EEVEE",
                                    object_name="flame_template",
                                    pkl_path=pkl_path,
                                    output_dir=binary_masks_dir,
                                )

                            # Save metadata
                            import json
                            inputs = {
                                'identity_idx': identity_idx,
                                'lighting_idx': lighting_idx,
                                'expression_idx': expression_idx,
                                'view_idx': view_idx,
                                'identity_seed': identity_seed,
                                'lighting_seed': lighting_seed,
                                'expression_seed': expression_seed,
                                'view_seed': view_seed,
                                'texture_file': cached_texture_path,
                                'envmap_file': cached_envmap_path,
                                'texture_path': texture_path,
                                'envmap_path': envmap_path,
                                'render_engine': render_engine,
                            }
                            inputs_file = os.path.join(out_dir_view, file_name + '_inputs.json')
                            with open(inputs_file, 'w') as f:
                                json.dump(inputs, f, indent=4)

                            # Restore camera for next view
                            restore_camera_state('Camera', cam_pose_backup, cam_intr_backup)

    elif num_images is None:
        seed_everything(seed)
        render_single_image(
            mesh_file=mesh_file,
            flame_shape_db=flame_shape_db,
            flame_expression_db=flame_expression_db,
            out_dir=out_dir,
            resolution_percentage=resolution_percentage,
            texture_path=texture_path,
            envmap_path=envmap_path,
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
            render_engine=render_engine,
            mask_type=mask_type,
            save_output_mesh=save_output_mesh,
        )
    else:
        ## back up the camera state so that we can restore it after each rendering 
        cam_pose_backup, cam_intr_backup = backup_camera_state('Camera')
        if start_index is not None and end_index is not None:
            image_rng = range(start_index, end_index + 1)
            assert end_index <= num_images, f"end_index {end_index} is out of bounds for num_images {num_images}"
        else:
            image_rng = range(num_images)
        print("Start index:", start_index)
        print("End index:", end_index)
        for i in image_rng:
            seed_everything(seed + i)
            out_dir_ = os.path.join(out_dir, f'image_{i:06d}')

            if Path(out_dir_).exists() and not overwrite:
                print(f"Output directory {out_dir_} already exists. Skipping...")
                continue

            render_single_image(
                mesh_file=mesh_file,
                flame_shape_db=flame_shape_db,
                flame_expression_db=flame_expression_db,
                out_dir=out_dir_,
                resolution_percentage=resolution_percentage,
                texture_path=texture_path,
                envmap_path=envmap_path,
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
                render_engine=render_engine,
                mask_type=mask_type,
                save_output_mesh=save_output_mesh,
            )
            ## restore the original camera state (so that the next randomization 
            ## happens wrt to the original camera and not the previous randomization)
            restore_camera_state('Camera', cam_pose_backup, cam_intr_backup)

    return




if __name__ == '__main__':
    main()
