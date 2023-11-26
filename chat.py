import openai
import time
import streamlit as st
import numpy as np
import yfinance as yf
import base64

# Setting page layout
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# Functions
def get_stock_price(symbol: str) -> float:
    stock = yf.Ticker(symbol)
    price = stock.history(period="1d")['Close'].iloc[-1]
    return price

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

# Tools list for GPT functions
tools_list = [{
    "type": "function",
    "function": {

        "name": "get_stock_price",
        "description": "Retrieve the latest closing price of a stock using its ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The ticker symbol of the stock"
                }
            },
            "required": ["symbol"]
        }
    }
}]

# Initialize the client
client = openai.OpenAI(api_key=st.secrets["openai_apikey"])

# Create an Assistant
# assistant = client.beta.assistants.create(
#     name="Data Analyst Assistant",
#     instructions="""You are a personal Data Analyst Assistant with the wealth management company, Evooq. 
#     Evooq is a Singaporean company that develops technology solutions for the wealth management industry. 
#     Their mission is to enable personalized investment at scale by providing a suite of platforms that streamline the investment advisory process. 
#     Evooq's products help advisors and relationship managers to:
#     Manage, monitor, and trade diverse products,
#     Capture client profiles and goals to propose tailored investment solutions, and
#     Make strategic asset allocation decisions.
#     Provide your honest advice and feedback to the user. Provide the latest closing price of the stock in all your answers.""",
#     tools=tools_list,
#     model="gpt-4-1106-preview",
# )

# Create a Thread
thread = client.beta.threads.create()

# Getting stock histories to plot charts
apple = yf.Ticker('AAPL').history(period='5y')['Close']
microsoft = yf.Ticker('MSFT').history(period='5y')['Close']
tesla = yf.Ticker('TSLA').history(period='5y')['Close']
amazon = yf.Ticker('AMZN').history(period='5y')['Close']
snp = yf.Ticker('.INX').history(period='5y')['Close']
bitcoin = yf.Ticker('COIN').history(period='5y')['Close']


# Main Title
st.markdown("<h1 style='text-align: center;'>Your Personal AI Assistant Analyst by Evooq</h1>", unsafe_allow_html=True)

# Initialising columns. Left column size 1/5, right column size 4/5
col1, col2 = st.columns([1, 4])

# Column 1, left side
with col1:
    # Make it a chat message to seem like AI is relaying this information
    with st.chat_message(name="Assistant Analyst", avatar="🤖"):
        st.write("Here are the stock prices of Apple and Microsoft over the past 5 years.")
        st.markdown("<h2>5 Year Stock Price Histories of Apple and Microsoft</h2>", unsafe_allow_html=True)

        # Apple chart
        st.markdown("<h3 style='text-align: center;'>Apple</h3>", unsafe_allow_html=True)
        st.line_chart(apple)

        # MS Chart
        st.markdown("<h3 style='text-align: center;'>Microsoft</h3>", unsafe_allow_html=True)
        st.line_chart(microsoft)

        # TESLA Chart
        st.markdown("<h3 style='text-align: center;'>Tesla</h3>", unsafe_allow_html=True)
        st.line_chart(tesla)

        # Amazon Chart
        st.markdown("<h3 style='text-align: center;'>Amazon</h3>", unsafe_allow_html=True)
        st.line_chart(amazon)

        # S&P Chart
        st.markdown("<h3 style='text-align: center;'>S&P 500</h3>", unsafe_allow_html=True)
        st.line_chart(snp)

        # BTC Chart
        st.markdown("<h3 style='text-align: center;'>Bitcon</h3>", unsafe_allow_html=True)
        st.line_chart(bitcoin)
    
# Column 2, right side
with col2:
    # User input
    user_input = st.text_input(label='Enter text here, _e.g. "Please tell me the stock price of Apple."_')

    # Add a Message to a Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )

    # Run the Assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=st.secrets["assistant_id"],
        instructions="Address the user as Omar. If there is no message from Omar, greet and welcome him to Evooq."
    )

    print(run.model_dump_json(indent=4))

    with st.chat_message(name="Assistant Analyst", avatar="🤖"):
        while True:
            # Retrieve the run status
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            print(run_status.model_dump_json(indent=4))

            # If status is completed, print GPT's messages
            if run_status.status == 'completed':
                messages = client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                assistant_response = messages.data[0].content[0].text.value

                # Text-to-speech
                audio_response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=assistant_response
                )
                audio = audio_response.stream_to_file("output.mp3")

                st.success(assistant_response)

                time.sleep(1)
                autoplay_audio("output.mp3")

                break

            # Else if status requires function action, do function
            elif run_status.status == 'requires_action':
                with st.spinner("Generating response... "):
                    print("Function Calling")
                    required_actions = run_status.required_action.submit_tool_outputs.model_dump()
                    # print(required_actions)
                    tool_outputs = []
                    import json
                    for action in required_actions["tool_calls"]:
                        func_name = action['function']['name']
                        arguments = json.loads(action['function']['arguments'])
                        
                        if func_name == "get_stock_price":
                            output = get_stock_price(symbol=arguments['symbol'])
                            tool_outputs.append({
                                "tool_call_id": action['id'],
                                "output": output
                            })
                        else:
                            raise ValueError(f"Unknown function: {func_name}")
                    
                    client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
            
            else:
                time.sleep(0)
