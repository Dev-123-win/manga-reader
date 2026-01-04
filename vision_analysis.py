import json
import base64
import io
from PIL import Image
import google.generativeai as genai
import os

# Gemini pricing is significantly lower than GPT-4o
# Flash 1.5 is ~$0.15 per million tokens for input
VISION_PRICE_PER_TOKEN = 0.00000015 
GPT_3_5_TURBO_PRICE_PER_TOKEN = 0.00000015

def get_gemini_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Please add it to your .env file.")
    genai.configure(api_key=api_key)
    # Using Pro for best summary quality, but Flash is also an option for speed/cost
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name)

def base64_to_pil(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

def analyze_images_with_gpt4_vision(
    character_profiles, pages, client, prompt, instructions, detail="low"
):
    """
    Analyzes manga pages using Gemini Vision.
    Kept the original function name to minimize changes in app.py.
    """
    model = get_gemini_model()
    
    # Construct contents for Gemini
    # We include instructions, profile references, and the sequence of pages
    contents = [
        f"System Instructions: {instructions}\n\n",
        "Here are some character profile pages, for your reference:",
    ]
    contents.extend([base64_to_pil(img) for img in character_profiles])
    contents.append(f"\nPrompt: {prompt}\n")
    contents.extend([base64_to_pil(img) for img in pages])
    
    response = model.generate_content(contents)
    
    # Mocking OpenAI response structure to minimize app.py changes
    class MockMessage:
        def __init__(self, content):
            self.content = content
    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)
    class MockUsage:
        def __init__(self, tokens):
            # Gemini counting is different, we'll return 0 for usage here 
            # as it doesn't affect the narration/video pipeline
            self.total_tokens = tokens
    class MockResponse:
        def __init__(self, content, tokens):
            self.choices = [MockChoice(content)]
            self.usage = MockUsage(tokens)
            
    return MockResponse(response.text, 0)

def detect_important_pages(
    profile_reference,
    chapter_reference,
    pages,
    client,
    prompt,
    instructions,
    detail="low",
):
    """Detect profile and chapter start pages using Gemini's JSON mode."""
    model = get_gemini_model()
    
    contents = [
        f"System Instructions: {instructions}\n\n",
        "Here are some character profile pages, for your reference:"
    ]
    contents.extend([base64_to_pil(img) for img in profile_reference])
    contents.append("\nHere are some chapter start pages, for your reference:")
    contents.extend([base64_to_pil(img) for img in chapter_reference])
    contents.append(f"\nPrompt: {prompt}\n")
    contents.extend([base64_to_pil(img) for img in pages])
    
    # Enable JSON mode for structured output
    response = model.generate_content(
        contents,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
        )
    )
    
    response_text = response.text
    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError:
        # Simple extraction if JSON mode fails for some reason
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed_response = json.loads(json_match.group())
        else:
            print("Failed to parse JSON from Gemini response.")
            raise

    return {
        "total_tokens": 0,
        "parsed_response": parsed_response.get("important_pages", []),
    }

def get_important_panels(
    profile_reference, panels, client, prompt, instructions, detail="low"
):
    """Identify key panels using Gemini."""
    model = get_gemini_model()
    
    contents = [
        f"System Instructions: {instructions}\n\n",
        "Here are some character profile pages, for your reference:"
    ]
    contents.extend([base64_to_pil(img) for img in profile_reference])
    contents.append(f"\nPrompt: {prompt}\n")
    contents.extend([base64_to_pil(img) for img in panels])
    
    try:
        response = model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            )
        )
        response_text = response.text
    except Exception as e:
        print(f"Gemini Error during panel identification: {e}")
        return {"total_tokens": 0, "parsed_response": []}

    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed_response = json.loads(json_match.group())
        else:
            parsed_response = {"important_panels": []}

    return {
        "total_tokens": 0,
        "parsed_response": parsed_response.get("important_panels", []),
    }
