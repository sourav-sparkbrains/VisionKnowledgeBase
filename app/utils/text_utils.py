VISION_PROMPT = """
Analyze this image and respond in this exact JSON format with no extra text:
{
  "caption": "one sentence describing what this image shows",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "image_type": "one of: screenshot, whiteboard, chart, diagram, photo, other"
}
"""

RAG_PROMPT = """
You are a visual knowledge assistant.
Answer the user's question based ONLY on the retrieved images below.
Always cite which image your answer comes from using its caption.
Be concise and factual.

User question: {query}

Retrieved images:
{context}

Answer:
"""