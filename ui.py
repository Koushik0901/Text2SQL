import streamlit as st
import pandas as pd
from engine import inference


def ui():
    st.markdown("# Text2SQL")
    st.markdown(
        "#### ***A Transformer model trained on WikiSQL dataset that accepts natural language as input and returns SQL Query as output.***"
    )
    st.markdown("# Examples")
    df = pd.read_csv("./utils/examples.csv")
    st.table(df)
    st.markdown("# Try it out:")
    input_text = st.text_input(
        label="Natural Language Question",
        value="What is Record, when Date is November 9?",
        max_chars=80,
    )
    output = inference("./txt2sql.pt", input_text)
    print(output)
    st.markdown(f"##### **PREDICTION:** ***{output}***")
    st.markdown("## [Code on GitHub](https://github.com/Koushik0901/Text2SQL)")
    st.markdown("")
    st.markdown(
        """# Connect with me
  [<img height="30" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />][github]
  [<img height="30" src="https://img.shields.io/badge/linkedin-blue.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />][LinkedIn]
  [<img height="30" src = "https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"/>][instagram]
  
  [github]: https://github.com/Koushik0901
  [instagram]: https://www.instagram.com/koushik_shiv/
  [linkedin]: https://www.linkedin.com/in/koushik-sivarama-krishnan/""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    ui()
