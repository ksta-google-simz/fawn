import gradio as gr
from utils.face_embedding import save_exclusion_faces, reset_exclusion_faces, load_exclusion_faces
from utils.anonymize_faces_in_image import anonymize_faces_in_image
from model_setup import pipe, generator, fa
from PIL import Image
import tempfile
import os

def register_face(image):
    if image is None:
        return "❌ 이미지를 업로드해주세요."
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image = image.convert("RGB")
        image.save(tmp.name)
        save_exclusion_faces([tmp.name])
    return "✅ 얼굴 등록 완료!"

def reset_faces():
    reset_exclusion_faces()
    return "♻️ 등록된 얼굴 초기화 완료!"

def show_faces():
    faces = load_exclusion_faces()
    return f"📦 등록된 얼굴 수: {len(faces)}"

def run_anonymizer(image):
    if image is None:
        return None, "❌ 이미지를 업로드해주세요."
    
    anon_image = anonymize_faces_in_image(
        image=image,
        face_alignment=fa,
        pipe=pipe,
        generator=generator,
        face_image_size=512,
        num_inference_steps=10,
        guidance_scale=4.0,
        anonymization_degree=1.25
    )
    return anon_image, "✅ 익명화 완료!"

with gr.Blocks() as demo:
    gr.Markdown("## 🕶️ 얼굴 익명화 시스템\n익명화 제외 인물을 등록하고, 나머지 얼굴만 익명화합니다.")

    with gr.Tab("1️⃣ 익명화 제외 인물 등록"):
        img = gr.Image(type="pil", label="익명화 제외할 얼굴")
        register_btn = gr.Button("등록")
        reset_btn = gr.Button("초기화")
        show_btn = gr.Button("등록된 얼굴 수 보기")
        register_output = gr.Textbox()
        show_output = gr.Textbox()
        reset_output = gr.Textbox()

        register_btn.click(fn=register_face, inputs=img, outputs=register_output)
        show_btn.click(fn=show_faces, outputs=show_output)
        reset_btn.click(fn=reset_faces, outputs=reset_output)

    with gr.Tab("2️⃣ 얼굴 익명화 실행"):
        anon_input = gr.Image(type="pil", label="익명화할 이미지")
        anon_btn = gr.Button("익명화 실행")
        result_img = gr.Image(type="pil", label="익명화 결과")
        result_text = gr.Textbox(label="결과 메시지")
        anon_btn.click(fn=run_anonymizer, inputs=anon_input, outputs=[result_img, result_text])

demo.launch(share=True)