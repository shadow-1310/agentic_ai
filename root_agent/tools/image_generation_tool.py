from google import genai
from google.genai import types
from google.adk.tools import ToolContext
import os
from PIL import Image

client = genai.Client(
    vertexai=True
)

async def generate_images(imagen_prompt: str, tool_context: ToolContext):
    print("************calling imagen model******************")
    try:
        response = client.models.generate_images(
            # model="imagen-4.0-generate-preview-06-06",
            model="imagen-3.0-generate-002",
            prompt=imagen_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="9:16",
                safety_filter_level="block_low_and_above",
                person_generation="allow_adult",
            ),
        )

        # print(response)
        generated_image_paths = []
        if response.generated_images is not None:
            for generated_image in response.generated_images:
                image_bytes = generated_image.image.image_bytes
                counter = str(tool_context.state.get("loop_iteration", 0))
                artifact_name = f"generated_image_" + counter + ".png"
                
                with open(artifact_name, "wb") as f:
                    f.write(image_bytes)
                report_artifact = types.Part.from_bytes(
                    data=image_bytes, mime_type="image/png"
                )
                try:
                    tool_context.save_artifact(artifact=report_artifact, filename=artifact_name)
                    artifact_list = tool_context.list_artifacts()
                    print("Artifacts in context:", artifact_list)
                    print(f"Image also saved as ADK artifact: {artifact_name}")
                except Exception as e:
                    print(f"error occured in saving artifacts:", e)

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

def save_to_gcs(tool_context: ToolContext, image_bytes, filename: str, counter: str):
    # --- Save to GCS ---
    storage_client = storage.Client()  # Initialize GCS client
    bucket_name = config.GCS_BUCKET_NAME

    unique_id = tool_context.state.get("unique_id", "")
    current_date_str = datetime.utcnow().strftime("%Y-%m-%d")
    unique_filename = filename
    gcs_blob_name = f"{current_date_str}/{unique_id}/{unique_filename}"

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_name)

    try:
        blob.upload_from_string(image_bytes, content_type="image/png")
        gcs_uri = f"gs://{bucket_name}/{gcs_blob_name}"

        # Store GCS URI in session context
        # Store GCS URI in session context
        tool_context.state["generated_image_gcs_uri_" + counter] = gcs_uri

    except Exception as e_gcs:

        # Decide if this is a fatal error for the tool
        return {
            "status": "error",
            "message": f"Image generated but failed to upload to GCS: {e_gcs}",
        }
        # --- End Save to GCS ---
