import gradio as gr
import tempfile
from PIL import Image
from main_interface import run_pipeline

def wrapped_pipeline(input_image, exclusion_image):
    # 입력 이미지 임시 파일 저장
    input_temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    input_image.save(input_temp.name)

    exclusion_path = None
    if exclusion_image is not None:
        exclusion_temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        exclusion_image = exclusion_image.convert("RGB")
        exclusion_image.save(exclusion_temp.name)
        exclusion_path = exclusion_temp.name

    try:
        result_path = run_pipeline(input_temp.name, exclusion_path)
        return result_path, "✅ 익명화가 완료되었습니다."
    except Exception as e:
        return None, f"❌ 오류 발생: {str(e)}"

iface = gr.Interface(
    fn=wrapped_pipeline,
    inputs=[
        gr.Image(type="pil", label="익명화할 이미지"),
        gr.Image(type="pil", label="익명화 제외할 인물 이미지 (선택)"),
    ],
    outputs=[
        gr.Image(label="익명화 결과"),
        gr.Text(label="처리 상태"),
    ],
    title="얼굴 익명화 시스템",
    description="얼굴을 익명화하되, 등록된 인물은 익명화 대상에서 제외합니다."
)
iface.launch(share=True)  # 코랩에서 링크 포함됨

if __name__ == "__main__":
    iface.launch(share=True)