from google import genai
from google.genai import types
from google.adk.tools import ToolContext
import os

# GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
# SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", 45))
# MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 1))
# IMAGEN_MODEL = os.getenv("IMAGEN_MODEL", "imagen-3.0-generate-002")
# GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-2.0-flash")


client = genai.Client(
    vertexai=True
)


async def generate_images(imagen_prompt: str, tool_context: ToolContext):

    try:

        response = client.models.generate_images(
            # model="imagen-3.0-generate-002",
            model="imagen-4.0-generate-preview-06-06",
            prompt=imagen_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="9:16",
                safety_filter_level="block_low_and_above",
                person_generation="allow_adult",
            ),
        )
        generated_image_paths = []
        if response.generated_images is not None:
            for generated_image in response.generated_images:
                # Get the image bytes
                image_bytes = generated_image.image.image_bytes
                counter = str(tool_context.state.get("loop_iteration", 0))
                artifact_name = f"generated_image_" + counter + ".png"
                report_artifact = types.Part.from_bytes(
                    data=image_bytes, mime_type="image/png"
                )

                await tool_context.save_artifact(artifact_name, report_artifact)
                print(f"Image also saved as ADK artifact: {artifact_name}")

                return {
                    "status": "success",
                    "message": f"Image generated .  ADK artifact: {artifact_name}.",
                    "artifact_name": artifact_name,
                }
        else:
            # model_dump_json might not exist or be the best way to get error details
            error_details = str(response)  # Or a more specific error field if available
            print(f"No images generated. Response: {error_details}")
            return {
                "status": "error",
                "message": f"No images generated. Response: {error_details}",
            }

    except Exception as e:

        return {"status": "error", "message": "No images generated.  {e}"}