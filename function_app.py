import azure.functions as func
import logging
import os
import langchain_openai
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="http_trigger_rag")
# This function will be triggered by an HTTP request
# It will fetch the relevant information from the database
# It will then use the Azure OpenAI model to generate the answer
# It will return the answer to the user
def http_trigger_rag(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get the environment variables
    endpoint = os.environ.get("endpoint")
    key = os.environ.get("key")
    credential = AzureKeyCredential(key)
    index_name = os.environ.get("index_name")
    azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.environ.get("OPENAI_API_KEY")
    azure_openai_embedding_deployment = os.environ.get("azure_openai_embedding_deployment")
    embedding_model_name = os.environ.get("embedding_model_name")
    azure_openai_api_version = os.environ.get("OPENAI_API_VERSION")

    # Get the Azure OpenAI credential
    # Get the token provider
    openai_credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(openai_credential, "https://cognitiveservices.azure.com/.default")

    # Init the Azure OpenAI client
    client = AzureOpenAI(
        azure_deployment=azure_openai_embedding_deployment,
        api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key,
        azure_ad_token_provider=token_provider if not azure_openai_key else None
    )

    # Init the Azure Search client
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    # Get the request body
    context = """"""
    req_body = req.get_json()

    # Get the embedding for the question
    embedding = client.embeddings.create(input=req_body['request'], model=embedding_model_name).data[0].embedding

    # Create a vector query for the question
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=50, fields="titleVector, snippetVector")

    # Get the top 20 documents related to the question
    # Get the title, snippet and link of the documents
    results = search_client.search(
        search_text=req_body['request'],
        vector_queries= [vector_query],
        select=["title", "snippet", "link"],
        top=20
    )

    # Create the context for the prompt
    # Append the title, snippet and link of the documents
    for result in results:
      context += f"Title: {result['title']}" + f"\nContent: {result['snippet']}" + f"\nLink: {result['link']}\n\n"


    # Create the prompt for the question
    # Append the context and the question
    qna_prompt_template = f"""You will be provided with the question and a related context, you need to answer the question using the context.

    Context:
    {context}

    Question:
    #{req_body['request']}

    Make sure to answer the question only using the context provided, if the context doesn't contain the answer then return "I don't have enough information to answer the question".

    Answer:"""
    
    # Get the answer using the Azure OpenAI model
    llm = langchain_openai.AzureOpenAI(deployment_name = "azure-llm-model",
                    model = "gpt-35-turbo-instruct",
                    temperature=1)
    
    # Get the response from the model
    response = llm(qna_prompt_template)

    # Create the output
    # Append the response and the context
    output = "Answer:\n" + response + "\n\nRelated documents:\n\n" + context
    
    # Return the output to the user
    return func.HttpResponse(output, status_code=200)