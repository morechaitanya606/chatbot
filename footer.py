import streamlit as st
from htbuilder import HtmlElement, div, a, p, img, styles
from htbuilder.units import percent, px

# Function to generate an image element with dynamic styling
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

# Function to create a hyperlink with styling
def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

# Function to build the layout and footer content
def layout(*args):
    # Defining the style to hide the menu and footer from Streamlit's default UI
    style = """
    <style>
        # MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp { bottom: 80px; }
        .st-emotion-cache-139wi93 {
            width: 100%;
            padding: 1rem 1rem 15px;
            max-width: 46rem;
        }
        .footer-container {
            background: linear-gradient(45deg, #ff6f61, #ff3f64);
            padding: 15px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            color: white;
            font-family: 'Arial', sans-serif;
            font-size: 16px;
            text-align: center;
            animation: fadeIn 1.5s ease-in-out;
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
    </style>
    """

    # Define a fixed footer style with modern visual enhancements
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        width=percent(100),
        color="white",
        text_align="center",
        height="auto",
        opacity=0.9,  # Slight opacity for visual effect
        padding=px(10, 20),
        font_size=px(14),
        box_shadow="0px 4px 8px rgba(0, 0, 0, 0.2)",
        border_radius=px(8),
    )

    # Create footer content with a modern design
    body = p()
    foot = div(style=style_div)(body)

    # Apply the custom style for hiding default Streamlit elements
    st.markdown(style, unsafe_allow_html=True)

    # Add dynamic content to footer
    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)

    # Render the footer content to the page with a nice background and animations
    st.markdown(str(foot), unsafe_allow_html=True)


# Function to render the footer with custom information
def footer():
    # Content for the footer - now with a modern design and vibrant colors
    footer_text = "Made with ❤️ by Chaitanya, Omkar, Kailas, Razin"
    layout(footer_text)


# Main entry point for the Streamlit app
if __name__ == "__main__":
    footer()  # Call footer to display it
