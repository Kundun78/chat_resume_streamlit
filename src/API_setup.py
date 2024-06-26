
from openai import OpenAI


#### EndPoint Prep ####

def set_model_endpoint(model_name):
    model_endpoints = {
        "meta-llama/Meta-Llama-3-8B-Instruct": ("http://llama38b.multivacplatform.org/v1", "multivac-FQ1cWX4DpshdhkXY2m"),
        "gradientai/Llama-3-8B-Instruct-262k": ("http://llama38b262k.multivacplatform.org/v1", "multivac-U8nH6PpNxvw2cXK9tM"),
        "mistralai/Mixtral-8x7B-Instruct-v0.1": ("http://mixtral8x7b.multivacplatform.org/v1", "multivac-4Q1cWX4DpshdhkXY2m"),
    }

    model_info = model_endpoints.get(model_name, "Model name not found")
    return model_name, model_info[0], model_info[1]
# set one of these models as the value to set_model_endpoint function



def client_openAI_init(my_model_name):
    # Set OpenAI's API key and API base to use vLLM's API server.



    model_name, openai_api_base, openai_api_key= set_model_endpoint(my_model_name)
    print(f"Model: {model_name}, Endpoint: {openai_api_base}, api_key: {openai_api_key}")

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client