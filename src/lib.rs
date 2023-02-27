

mod texture;
mod camera;
mod model;
mod resources;

use model::{Vertex,Model};
use crate::model::DrawModel;
use camera::*;
use cgmath::prelude::*;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    window::Window,
};


pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state=State::new(&window).await;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => if !state.input(event){
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    
                    WindowEvent::Resized(physical_size)=>{
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged {new_inner_size, .. }=>{
                        state.resize(**new_inner_size);
                    }
                
                    _ => {}
                }
            },

            Event::RedrawRequested(window_id)if window_id == window.id()=>{
                state.update();
                match state.render(){
                    Ok(_)=>{}
                    Err(wgpu::SurfaceError::Lost)=>state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory)=>*control_flow=ControlFlow::Exit,
                    Err(e)=>eprintln!("{:?}",e),
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }

            _ => {}
        }
    });
}






#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}
impl InstanceRaw{
    fn desc<'a>()->wgpu::VertexBufferLayout<'a>{
        use std::mem;
        wgpu::VertexBufferLayout{
            array_stride:mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode:wgpu::VertexStepMode::Instance,
            attributes:&[
                wgpu::VertexAttribute{
                    offset:0,
                    shader_location:5,
                    format:wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute{
                    offset: mem::size_of::<[f32;4]>() as wgpu::BufferAddress,
                    shader_location:6,
                    format:wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute{
                    offset: mem::size_of::<[f32;8]>() as wgpu::BufferAddress,
                    shader_location:7,
                    format:wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute{
                    offset: mem::size_of::<[f32;12]>() as wgpu::BufferAddress,
                    shader_location:8,
                    format:wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[allow(dead_code)]
const ROTATION_SPEED: f32 = 2.0 * std::f32::consts::PI / 60.0;

struct Instance{
    position:cgmath::Vector3<f32>,
    rotation:cgmath::Quaternion<f32>,
}
impl Instance{
    fn to_raw(&self)->InstanceRaw{

        InstanceRaw { 
            model: (cgmath::Matrix4::from_translation(self.position)*cgmath::Matrix4::from(self.rotation)).into(), 
        }
    }
}

const NUM_INSTANCES_PER_ROW: u32 = 10;



#[allow(dead_code)]
struct State{
    surface:wgpu::Surface,
    device:wgpu::Device,
    queue:wgpu::Queue,
    config:wgpu::SurfaceConfiguration,
    size:winit::dpi::PhysicalSize<u32>,
    render_pipeline:wgpu::RenderPipeline,
    diffuse_bind_group:wgpu::BindGroup,
    diffuse_texture:texture::Texture,
    camera:Camera,
    camera_controller:CameraController,
    camera_uniform:CameraUniform,
    camera_buffer:wgpu::Buffer,
    camera_bind_group:wgpu::BindGroup,
    instances:Vec<Instance>,
    instance_buffer:wgpu::Buffer,
    depth_texture:texture::Texture,
    obj_model:Model,
}

impl State{
    async fn new(window:&Window)->Self{
        //ウィンドウサイズの取得
        let size=window.inner_size();
        //バックエンドインスタンスの生成
        let instance=wgpu::Instance::new(wgpu::Backends::all());
        //画面の形成(unsafe code)
        let surface=unsafe{instance.create_surface(window)};
        //アダプター、GPUのハンドラを取得する
        let adapter=instance.request_adapter(
            &wgpu::RequestAdapterOptionsBase { 
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface:Some(&surface),
            },
        ).await.unwrap();
        //アダプターからデバイス(GPU)の設定を行う
        let (device,queue)=adapter.request_device(
            &wgpu::DeviceDescriptor{
                features:wgpu::Features::empty(),
                limits:if cfg!(target_arch="wasm32"){
                    wgpu::Limits::downlevel_webgl2_defaults()
                }else{
                    wgpu::Limits::default()
                },
                label:None,
            },
            None,
        ).await.unwrap();
        //Surfaceのコンフィグ設定
        let config=wgpu::SurfaceConfiguration{
            usage:wgpu::TextureUsages::RENDER_ATTACHMENT,
            format:surface.get_supported_formats(&adapter)[0],
            width:size.width,
            height:size.height,
            present_mode:wgpu::PresentMode::Fifo,
            alpha_mode:wgpu::CompositeAlphaMode::Auto,
        };
        //コンフィグ変更の確定
        surface.configure(&device,&config);


        //シェーダーファイルの読み込み
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label:Some("Shader"),
            source:wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        //テクスチャバインドグループの設定
        let texture_bind_group_layout=
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
                entries:&[
                    wgpu::BindGroupLayoutEntry{
                        binding:0,
                        visibility:wgpu::ShaderStages::FRAGMENT,
                        ty:wgpu::BindingType::Texture { 
                            sample_type: wgpu::TextureSampleType::Float { filterable: true }, 
                            view_dimension: wgpu::TextureViewDimension::D2, 
                            multisampled: false,
                        },
                        count:None,
                    },
                    wgpu::BindGroupLayoutEntry{
                        binding:1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty:wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count:None,
                    },
                ],
                label:Some("texture_bind_group_layout"),
            });

        let depth_texture=texture::Texture::create_depth_texture(&device, &config, "depth_texture");
        
        //カメラバインドグループの設定
        let camera_bind_group_layout=device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            entries:&[
                wgpu:: BindGroupLayoutEntry{
                    binding:0,
                    visibility:wgpu::ShaderStages::VERTEX,
                    ty:wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Uniform, 
                        has_dynamic_offset: false, 
                        min_binding_size: None,
                    },
                    count:None
                }
            ],
            label:Some("camera_bind_group_layout"),
        });

        //レンダーパイプラインの定義
        let render_pipeline_layout=
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
                label:Some("Render Pipeline Layout"),
                bind_group_layouts:&[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                ],
                push_constant_ranges:&[],
        });

        //レンダーパイプラインの編集
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor{
            label:Some("Render Pipeline"),
            layout:Some(&render_pipeline_layout),
            //頂点データの保存
            vertex:wgpu::VertexState{
                module:&shader,
                entry_point:"vs_main",
                buffers:&[
                    model::ModelVertex::desc(),
                    InstanceRaw::desc()
                ],
            },
            //色データの保存に使う
            fragment:Some(wgpu::FragmentState{
                module:&shader,
                entry_point:"fs_main",
                targets:&[Some(wgpu::ColorTargetState{
                    format:config.format,
                    blend:Some(wgpu::BlendState{
                        color:wgpu::BlendComponent::REPLACE,
                        alpha:wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask:wgpu::ColorWrites::ALL,
                })],
            }),
            //頂点からポリゴンを形成するときの頂点解釈の設定
            primitive:wgpu::PrimitiveState{
                topology:wgpu::PrimitiveTopology::TriangleList,
                strip_index_format:None,
                front_face:wgpu::FrontFace::Ccw,
                cull_mode:Some(wgpu::Face::Back),
                polygon_mode:wgpu::PolygonMode::Fill,
                unclipped_depth:false,
                conservative:false,
            },

            //深度バッファの使用設定
            depth_stencil:Some(wgpu::DepthStencilState{
                format:texture::Texture::DEPTH_FORMAT,
                depth_write_enabled:true,
                depth_compare:wgpu::CompareFunction::Less,
                stencil:wgpu::StencilState::default(),
                bias:wgpu::DepthBiasState::default(),
            }),
            //マルチサンプリングの設定
            multisample:wgpu::MultisampleState{
                count:1,
                mask:!0,
                alpha_to_coverage_enabled:false,
            },
            multiview:None,

        });


        //テクスチャ******************************
        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture=texture::Texture::from_bytes(&device,&queue,diffuse_bytes,"happy-tree.png").unwrap();
        let diffuse_bind_group=device.create_bind_group(
            &wgpu::BindGroupDescriptor{
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry{
                        binding:0,
                        resource:wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry{
                        binding:1,
                        resource:wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    }
                ],
                label: Some("diffuse_bind_group"),
            }
        );
        //*****************************************/


        //カメラ設定
        let camera=Camera { 
            eye: (0.0,1.0,2.0).into(), 
            target: (0.0,0.0,0.0).into(), 
            up:cgmath::Vector3::unit_y(),
            aspect: config.width as f32/ config.height as f32,
            fovy: 45.0,
            znear: 0.1, 
            zfar: 100.0, 
        };

        let camera_controller =CameraController::new(0.2);

        let mut camera_uniform=CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer=device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor{
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage:wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST,
            }
        );

        

        let camera_bind_group=device.create_bind_group(&wgpu::BindGroupDescriptor{
            layout:&camera_bind_group_layout,
            entries:&[
                wgpu::BindGroupEntry{
                    binding:0,
                    resource:camera_buffer.as_entire_binding(),
                }
            ],
            label:Some("camera_bind_group"),
        });

        const SPACE_BETWEEN: f32 = 3.0;
        let instances=(0..NUM_INSTANCES_PER_ROW).flat_map(|z|{
            (0..NUM_INSTANCES_PER_ROW).map(move |x|{
                let x=SPACE_BETWEEN*(x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z=SPACE_BETWEEN*(z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = cgmath::Vector3{x,y:0.0,z};

                let rotation = if position.is_zero(){
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                }else{
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };

                Instance{
                    position,rotation,
                }
            })
        }).collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer= device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor{
                label:Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage:wgpu::BufferUsages::VERTEX|wgpu::BufferUsages::COPY_DST,
            }
        );

        let obj_model = resources::load_model(
            "cube.obj",
            &device,
            &queue,
            &texture_bind_group_layout,
        ).await.unwrap();

        //戻り値
        Self{
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            diffuse_bind_group,
            diffuse_texture,
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            instances,
            instance_buffer,
            depth_texture,
            obj_model,
        }

    }

    //ウィンドウサイズの変更時、Surface側のconfig内にある画面サイズの更新を行うための関数
    fn resize(&mut self,new_size:winit::dpi::PhysicalSize<u32>){
        if new_size.width>0 &&new_size.height>0{
            self.size=new_size;
            self.config.width=new_size.width;
            self.config.height=new_size.height;
            self.depth_texture=texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.surface.configure(&self.device,&self.config);
        }
    }
    

    //入力を受け付けるためのもの、ここがTrueだとrunループ側の処理が無効化される
    fn input(&mut self,event:&WindowEvent)->bool{

        self.camera_controller.process_events(event)
    }
    fn update(&mut self){
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
        
        
        //spining cord
        /*
        
        for instance in &mut self.instances {
            let amount = cgmath::Quaternion::from_angle_y(cgmath::Rad(ROTATION_SPEED));
            let current = instance.rotation;
            instance.rotation = amount * current;
        }
        let instance_data = self
            .instances
            .iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&instance_data),
        );
        */
    }

    fn render(&mut self)->Result<(),wgpu::SurfaceError>{
        let output=self.surface.get_current_texture()?;
        let view=output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        //GPUに送信するための命令コードを生成するためのエンコーダを作成する
        let mut encoder=self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
            label:Some("Render Encoder"),
        });


        //このコードブロックは生成したエンコーダを一次的に借用し、使用後に借用を開放するためにおいてあるものである
        //begin_render_pass()はRustの仕様に従い、この時点でencoderの借用権限を保有する
        //ここで借用の返却をおこなわいとencoder.finish()を実行することができなくなってしまう
        {
            let mut render_pass=encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
                label:Some("Render Pass"),
                color_attachments:&[Some(wgpu::RenderPassColorAttachment{
                    view:&view,
                    resolve_target:None,
                    ops:wgpu::Operations{
                        load:wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store:true,
                    },
                })],
                depth_stencil_attachment:Some(wgpu::RenderPassDepthStencilAttachment{
                    view:&self.depth_texture.view,
                    depth_ops:Some(wgpu::Operations{
                        load:wgpu::LoadOp::Clear(1.0),
                        store:true,
                    }),
                    stencil_ops:None,
                }),
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.draw_model_instanced(&self.obj_model, 0..self.instances.len() as u32,&self.camera_bind_group);
        }

        //コマンドバッファの編集を終了し、コマンドをGPUに送信する(encoderの借用の開放を忘れずに！)
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())

    }
}