import chess
import chess.svg
import json
import urllib.request
import webbrowser
import os
import time
import random

def llm_move(board, model_name):
    url = "http://localhost:20128/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY_HERE"
    }
    legal = [m.uci() for m in board.legal_moves]
    prompt = f"FEN: {board.fen()}\nLegal: {legal}\nReply EXACTLY with ONE UCI move from list. No extra text."
    
    data = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": False
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, headers=headers)
    
    print("\n" + "="*40)
    print(f"[DEBUG] MODEL: {model_name}")
    print(f"[DEBUG] PROMPT:\n{prompt}")
    print("="*40)
    
    with urllib.request.urlopen(req) as res:
        raw_res = res.read()
        print(f"[DEBUG] RAW RESP:\n{raw_res.decode('utf-8')}")
        print("="*40)
        body = json.loads(raw_res)
        return body["choices"][0]["message"]["content"].strip()

def render_gui(board, white_name, black_name):
    html = f"""
    <html>
    <head>
        <meta http-equiv="refresh" content="2">
        <style>
            body {{ display:flex; flex-direction:column; justify-content:center; align-items:center; background:#222; color:#fff; font-family:sans-serif; height:100vh; margin:0; }}
            .player {{ font-size: 24px; font-weight: bold; margin: 15px; padding: 10px 20px; background: #444; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="player">Black: {black_name}</div>
        {chess.svg.board(board, size=500)}
        <div class="player">White: {white_name}</div>
    </body>
    </html>
    """
    with open("ui_preview.html", "w") as f:
        f.write(html)

def main():
    print("=== AI Chess Arena ===")
    print("Select Game Mode:")
    print("1. Player vs Player")
    print("2. Player vs LLM")
    print("3. LLM vs LLM")
    
    mode_choice = input("Enter mode (1-3): ").strip()
    
    white_is_human = True
    black_is_human = True
    white_name = "Player 1"
    black_name = "Player 2"
    
    if mode_choice == "2":
        print("\nWho plays White?")
        print("1. Player")
        print("2. LLM")
        color_choice = input("Enter choice (1-2): ").strip()
        if color_choice == "1":
            white_is_human = True
            black_is_human = False
            white_name = "Human"
            black_name = input("Enter LLM model name for Black (e.g., model-black): ").strip() or "model-black"
        else:
            white_is_human = False
            black_is_human = True
            white_name = input("Enter LLM model name for White (e.g., model-white): ").strip() or "model-white"
            black_name = "Human"
            
    elif mode_choice == "3":
        white_is_human = False
        black_is_human = False
        white_name = input("Enter LLM model name for White: ").strip() or "model-white"
        black_name = input("Enter LLM model name for Black: ").strip() or "model-black"

    board = chess.Board()
    render_gui(board, white_name, black_name)
    webbrowser.open("file://" + os.path.realpath("ui_preview.html"))

    while not board.is_game_over():
        is_white_turn = (board.turn == chess.WHITE)
        current_is_human = white_is_human if is_white_turn else black_is_human
        current_name = white_name if is_white_turn else black_name
        
        print(f"\n{current_name}'s turn ({"White" if is_white_turn else "Black"}).")
        
        if current_is_human:
            uci = input("Enter move (UCI format, e.g., e2e4): ").strip()
        else:
            print(f"{current_name} is thinking...")
            try:
                uci = llm_move(board, current_name)
                print(f"Move: {uci}")
            except Exception as e:
                print(f"Error: {e}. Forcing random move.")
                uci = random.choice(list(board.legal_moves)).uci()

        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                board.push(move)
                render_gui(board, white_name, black_name)
            else:
                if not current_is_human:
                    print("Illegal move. Forcing random move to prevent freeze.")
                    board.push(random.choice(list(board.legal_moves)))
                    render_gui(board, white_name, black_name)
                else:
                    print("Illegal move. Try again.")
        except Exception:
            if current_is_human:
                print("Invalid format. Use UCI format like 'e2e4'. Try again.")
            else:
                print("Invalid LLM response. Forcing random move.")
                board.push(random.choice(list(board.legal_moves)))
                render_gui(board, white_name, black_name)
                
        if not current_is_human:
            time.sleep(1)

    print("\nGame Over:", board.result())

if __name__ == "__main__":
    main()
