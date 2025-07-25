from google.cloud import texttospeech
import os

def synthesize_text(text, output_filename="output.mp3"):
    """Synthesizes speech from the input text and saves it to a file."""

    # Initialize the client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Configure voice parameters
    # You can explore available voices at:
    # https://cloud.google.com/text-to-speech/docs/voices
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL  # Or FEMALE, MALE
        # name="en-US-Wavenet-C" # For a specific Wavenet voice
    )

    # Set audio configuration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3  # Or LINEAR16, OGG_OPUS
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Write the binary audio content to a local file
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
    print(f"Audio content written to file '{output_filename}'")

if __name__ == "__main__":
    text_to_convert = "The name of Chapter 3 of the Class 6 English textbook is Nurturing Nature."
    synthesize_text(text_to_convert)