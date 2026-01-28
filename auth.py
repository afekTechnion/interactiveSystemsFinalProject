import sqlite3
import bcrypt
import streamlit as st
import os
import re

# configurations
BASE_DB_FOLDER = "Database"
USERS_DB_FILE = os.path.join(BASE_DB_FOLDER, "users.db")

# ensure DB folder exists
if not os.path.exists(BASE_DB_FOLDER):
    os.makedirs(BASE_DB_FOLDER)


# initialize user database
def init_user_db():
    conn = sqlite3.connect(USERS_DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password BLOB
        )
    ''')
    conn.commit()
    conn.close()


def validate_password(password):
    """
    Enforces password complexity:
    1. At least 6 characters long
    2. Contains at least one digit
    """
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number."
    return True, ""


def hash_password(password):
    """Securely hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


def check_password(password, hashed_pw):
    """Checks a password against its hash."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_pw)
    except ValueError:
        return False


# database operations
def add_user(username, password):
    conn = sqlite3.connect(USERS_DB_FILE)
    c = conn.cursor()
    hashed_pw = hash_password(password)

    # try to insert the new user
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
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()

    if data:
        stored_hash = data[0]
        if check_password(password, stored_hash):
            return True

    return False


# UI rendering
def render_login_ui():
    st.markdown("""
        <style>
            .block-container { padding-top: 3rem; padding-bottom: 2rem; }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col2:
        st.markdown("<h1 style='text-align: center;'>🎬 PinPoint</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; color: grey; margin-bottom: 30px;'>Sign in to access your knowledge base</p>",
            unsafe_allow_html=True)

        with st.container(border=True):
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

            # login tab
            with tab1:
                st.write("")
                # login form
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type='password')

                    st.write("")
                    submit_login = st.form_submit_button("Log In", type="primary", use_container_width=True)

                if submit_login:
                    if login_user(username, password):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.toast(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Incorrect Username or Password")

            # signup tab
            with tab2:
                st.write("")
                with st.form("signup_form"):
                    new_user = st.text_input("Choose a Username")
                    new_password = st.text_input("Choose a Password", type='password')

                    st.write("")
                    submit_signup = st.form_submit_button("Create Account", use_container_width=True)

                if submit_signup:
                    # check if empty
                    if not new_user or not new_password:
                        st.warning("Please fill in all fields.")

                    # check password restrictions
                    else:
                        is_valid, msg = validate_password(new_password)
                        if not is_valid:
                            st.error(msg)
                        else:
                            # try to add user
                            if add_user(new_user, new_password):
                                st.success("Account created! Switch to Login tab.")
                            else:
                                st.warning("That username is taken.")
