## FrontEnd
    - Build UI - vibe code
    It should have the design as the screenshot
    It should have one history component
    Then it should pass the question to the chatbot function for now lets make it dummy - but the response will be in the format of modelResponse, and modelResponse should have two component Timeline, SkillCard, ProjectCard(With url clickable if any) so we based on the enum we should render the component, so the answer should show both the text and render the component in the same ansswer component

## BackEnd
    - Build GuardRail Agent: Should have clear prompt and structured output to tell about the question is related to the portfolio or not

    - Should create Langgraph conditional router if the question is not related to the portfolio then we should response out the default outputs (Better to have 5 random default outputs in a list and randomize it)

    - History file: Store all the question using the uniqueSessionId created for that particular user along with the question store also the validation status of the question
    
    - Portfolio data file: It should have all the data about me in a structured manner 

    - Embedded File: Portfolio data have to be embedded properly and should be stored in a vector database

    - Embedding Agent: It should embed the question and get the relevant documents from the embedded file and pass it along with the question to the next llm agent

    - Demolisher: This agent should analyze the question with the retrieved documents and should remove all the irrelavent retrieved documents

    - Should have another node in langgraph if the retrived document list is empty then we should consider default outputs

    - Build LLM Agent: It should answer the question it gets using the retrieved content 


## Schema:
    # StateSchema
        - user_question: str
        - is_question_porfolio_related: boolean
        - retrieved_docs: []
        - output: ModelResponse

    # ModelResponse:
        - text_content: str
        - has_ui_render_component: enum(TimeLine, skillCard, projectCard)
        - ui_component: json

## Tools
- FAISS
- Jsonbin.io  - to store history file
- HuggingFace
- Plain Js for frontend
- Langgraph, Langchain

