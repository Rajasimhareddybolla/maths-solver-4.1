import streamlit as st
from pathlib import Path
import sys
import os
import dotenv
# Ensure src is importable
sys.path.append('src')

dotenv.load_dotenv()

with open('.openai_api_key','w') as file:
    file.write(os.environ["OPENAI_API_KEY"])


from src.math_solver.app import create_app

# Create app once
@st.cache_resource
def get_solver_app():
    return create_app()

app = get_solver_app()

st.set_page_config(page_title="Math Solver — Streamlit", layout="wide")

st.title("Math Solver")
st.markdown("A neat front-end for the Math Solver package. Enter a question and optionally add images.")

with st.sidebar:
    st.header("Options")
    show_stats = st.checkbox("Show statistics", value=True)
    ignore_history = st.checkbox("Ignore conversation history", value=False)

# Main input area
st.subheader("Problem input")
question = st.text_area("Enter the math problem text", height=120)

st.markdown("**Image inputs (optional)**")
uploaded_files = st.file_uploader("Upload images (these will be saved temporarily)", type=["png","jpg","jpeg"], accept_multiple_files=True)
image_paths_text = st.text_area("Or provide image paths (comma separated)", value="", help="Paths local to this environment, e.g. tests/qs1.png")

col1, col2 = st.columns([2,1])

with col1:
    if st.button("Solve"):
        if not question.strip() and not (uploaded_files or image_paths_text.strip()):
            st.warning("Please provide a question or images to solve.")
        else:
            # Save uploaded files to a temporary directory
            temp_paths = []
            tmp_dir = Path(".streamlit_uploads")
            tmp_dir.mkdir(exist_ok=True)
            for uf in uploaded_files:
                dest = tmp_dir / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
                temp_paths.append(str(dest))

            # Parse text image paths
            text_paths = [p.strip() for p in image_paths_text.split(",") if p.strip()]
            image_paths = temp_paths + text_paths if (temp_paths or text_paths) else None

            with st.spinner("Solving problem — this may take a few seconds..."):
                try:
                    res = app.solve_problem(question or None, image_paths=image_paths, ignore_history=ignore_history)
                except Exception as e:
                    st.error(f"Error while solving: {e}")
                    res = None

            if res:
                st.success("Solved")
                st.markdown("---")
                st.subheader("Solution (structured)")

                # Show main solution and formatted output
                st.json({
                    "conversation_id": res.get("conversation_id"),
                    "success": res.get("success"),
                    "processing_time": res.get("processing_time"),
                    "images_processed": res.get("images_processed"),
                    "history_ignored": res.get("history_ignored"),
                })

                st.markdown("**Formatted Output**")
                formatted = res.get("formatted_output") or res.get("solution")
                if isinstance(formatted, dict) or isinstance(formatted, list):
                    st.json(formatted)
                else:
                    st.code(str(formatted))

                st.markdown("**Token usage**")
                st.table(res.get("token_usage") or {})

                st.markdown("**Input information**")
                st.json(res.get("input_information") or {})

with col2:
    st.markdown("## Usage & Performance")
    if show_stats:
        try:
            stats = app.get_statistics()
            if not stats:
                st.info("No statistics available.")
            else:
                # Top cards
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### Conversations")
                    st.markdown(f"**{stats.get('total_conversations', 0)}** total")
                    st.caption(f"Successful: {stats.get('successful_conversations', 0)}  •  Failed: {stats.get('failed_conversations', 0)}")
                    # success rate progress
                    total = stats.get('total_conversations', 0) or 1
                    success = stats.get('successful_conversations', 0)
                    success_rate = int((success / total) * 100)
                    st.progress(success_rate)

                with c2:
                    st.markdown("### Tokens")
                    st.metric("Total tokens", stats.get('total_tokens', 0))
                    st.metric("Avg / conversation", round(stats.get('average_tokens_per_conversation', 0), 2))

                # Small bar chart for token breakdown if available
                token_breakdown = {
                    'prompt_tokens': stats.get('total_prompt_tokens', 0),
                    'completion_tokens': stats.get('total_completion_tokens', 0)
                }
                try:
                    import pandas as _pd
                    tb_df = _pd.DataFrame.from_dict(token_breakdown, orient='index', columns=['tokens'])
                    st.markdown("### Token breakdown")
                    st.bar_chart(tb_df)
                except Exception:
                    # Fallback to simple JSON
                    st.json(token_breakdown)

                # Raw JSON collapsed
                with st.expander("Raw statistics JSON"):
                    st.json(stats)
        except Exception as e:
            st.error(f"Failed to retrieve statistics: {e}")

st.markdown("---")
st.markdown("Small note: Uploaded images are temporarily saved in `.streamlit_uploads` in the repo root.")
