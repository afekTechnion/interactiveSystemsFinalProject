import sqlite3
import hashlib
import streamlit as st

# Constants
USER_DB_PATH = "users.db"

def init_user_db():
    # Initialize the user database if it does not exist
    conn = sqlite3.connect(USER_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    # Hash password using SHA256 for basic security
    return hashlib.sha256(str.encode(password)).hexdigest()

def add_user(username, password):
    # Add a new user to the database
    conn = sqlite3.connect(USER_DB_PATH)
    c = conn.cursor()
    hashed_pw = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def check_login(username, password):
    # Verify user credentials
    conn = sqlite3.connect(USER_DB_PATH)
    c = conn.cursor()
    hashed_pw = hash_password(password)
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_pw))
    data = c.fetchall()
    conn.close()
    return len(data) > 0

def render_auth_ui():
    # Render the sidebar login/signup forms
    st.sidebar.title("Authentication")
    choice = st.sidebar.radio("Login / Signup", ["Login", "Signup"])

    if choice == "Signup":
        st.sidebar.subheader("Create New Account")
        new_user = st.sidebar.text_input("Username", key="signup_user")
        new_password = st.sidebar.text_input("Password", type="password", key="signup_pass")
        if st.sidebar.button("Signup"):
            if add_user(new_user, new_password):
                st.sidebar.success("Account Created! Please Login.")
            else:
                st.sidebar.error("Username already exists.")

    elif choice == "Login":
        st.sidebar.subheader("Login Section")
        username = st.sidebar.text_input("Username", key="login_user")
        password = st.sidebar.text_input("Password", type="password", key="login_pass")
        if st.sidebar.button("Login"):
            if check_login(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.sidebar.error("Incorrect Username or Password")