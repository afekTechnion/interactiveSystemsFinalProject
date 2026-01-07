import sqlite3
import hashlib
import streamlit as st
import os

# Configuration
BASE_DB_FOLDER = "Database"
USERS_DB_FILE = os.path.join(BASE_DB_FOLDER, "users.db")

# Ensure DB folder exists
if not os.path.exists(BASE_DB_FOLDER):
    os.makedirs(BASE_DB_FOLDER)


def init_user_db():
    conn = sqlite3.connect(USERS_DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False


def add_user(username, password):
    conn = sqlite3.connect(USERS_DB_FILE)
    c = conn.cursor()
    hashed_pw = make_hashes(password)
    try:
        c.execute('INSERT INTO users(username, password) VALUES (?,?)', (username, hashed_pw))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success


def login_user(username, password):
    conn = sqlite3.connect(USERS_DB_FILE)
    c = conn.cursor()
    hashed_pw = make_hashes(password)
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_pw))
    data = c.fetchall()
    conn.close()
    return data


def render_login_ui():
    # 1. Custom CSS to clean up the top padding and center things nicely
    st.markdown("""
        <style>
            .block-container {
                padding-top: 3rem;
                padding-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # 2. Use columns to center the content (Left Spacer | Content | Right Spacer)
    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col2:
        # Header with centered text and emoji
        st.markdown("<h1 style='text-align: center;'>🎬 PinPoint</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; color: grey; margin-bottom: 30px;'>Sign in to access your knowledge base</p>",
            unsafe_allow_html=True)

        # 3. Use a Container with a border to create a "Card" look
        with st.container(border=True):
            # Add icons to tabs for better visuals
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

            # --- Login Tab ---
            with tab1:
                st.write("")  # Add a little breathing room
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type='password', key="login_pass")

                st.write("")
                # use_container_width=True makes the button stretch to match inputs
                # type="primary" makes it red/bold (the theme color)
                if st.button("Log In", use_container_width=True, type="primary"):
                    if login_user(username, password):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.toast(f"Welcome back, {username}!")  # Nice popup effect
                        st.rerun()
                    else:
                        st.error("Incorrect Username or Password")

            # --- Sign Up Tab ---
            with tab2:
                st.write("")
                new_user = st.text_input("Choose a Username", key="signup_user")
                new_password = st.text_input("Choose a Password", type='password', key="signup_pass")

                st.write("")
                if st.button("Create Account", use_container_width=True):
                    if add_user(new_user, new_password):
                        st.success("Account created! Please switch to Login tab.")
                    else:
                        st.warning("That username is taken.")
