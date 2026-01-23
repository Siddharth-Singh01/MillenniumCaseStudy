
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool, tool



# model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key="e67aabc3-4d2c-47b7-83bd-b3fd4ccc40be",
model = ChatOpenAI(model="gpt-5o-mini", temperature=0, api_key="",
 base_url="")

prompt = ChatPromptTemplate.from_template(
    "You are a product categorizer. Choose the single best category from this list:\n"
    "{categories}\n\n"
    "Rules:\n"
    "- Answer with only the category text, nothing else.\n"
    "- If unsure, pick the closest category.\n"
    "Item: {item}\n"
    "Category:"

)
CATEGORIES = ["Electronics", "Fashion", "Home", "Groceries", "Books", "Toys", "Beauty", "Sports"]

chain_ = prompt | model | StrOutputParser()
def categorize_item_with_prompt(item_name: str) -> str:
    return chain_.invoke({
        "item": item_name,
        "categories": ", ".join(CATEGORIES)
    })

@tool
def list_categories() -> str:
    """Return the allowed product categories as a comma-separated string."""
    return ", ".join(CATEGORIES)

def categorize_item_with_tool(item_name: str) -> str:
    system = SystemMessage(content="You are a product categorizer. Use tools when helpful. Output only one category.")
    model_with_tools = model.bind_tools([list_categories])
    response = model_with_tools.invoke([
        system,
        HumanMessage(content=f"Item: {item_name}\nCategory:")
    ])
    return str(response.content).strip()


if __name__ == "__main__":
    print(categorize_item_with_prompt("iPhone 15 Pro case"))       # -> Electronics
    print(categorize_item_with_prompt("Organic almond milk"))      # -> Groceries
    print(categorize_item_with_prompt("Foam yoga mat"))            # -> Sports
    # Tool demo (direct invocation)
    print(list_categories.invoke({}))                                # -> Electronics, Fashion, ...
    # Tool demo (model-bound)
    print(categorize_item_with_tool("Kindle Paperwhite"))           # -> Books
    print(categorize_item_with_tool("iPhone 15 Pro case"))           # -> Books
    # Example multimodal call (image URL must be accessible to the model)
    # print(categorize_image(
    #     "https://upload.wikimedia.org/wikipedia/commons/3/3a/IPhone_15_Pro.png",
    #     hint="phone"
    # ))
