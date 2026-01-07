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
    st.title("Login System")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.subheader("Login Section")
        username = st.text_input("User Name")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if login_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success(f"Logged In as {username}")
                st.rerun()
            else:
                st.error("Incorrect Username or Password")

    with tab2:
        st.subheader("Create New Account")
        new_user = st.text_input("New Username")
        new_password = st.text_input("New Password", type='password')
        if st.button("Sign Up"):
            if add_user(new_user, new_password):
                st.success("Account created! Please Login.")
            else:
                st.warning("Username already exists")
