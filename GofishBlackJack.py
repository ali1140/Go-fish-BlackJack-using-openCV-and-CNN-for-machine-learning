import cv2 as cv
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pygame
import os
import sys

pygame.init()
window_width = 1000
window_height = 800
awal_game = True
deck_bot = []
deck_pemain = []
background_image = pygame.image.load("background.jpg")
background_image = pygame.transform.scale(background_image, (window_width, window_height))
screen = pygame.display.set_mode((window_width, window_height))
button_rects = []
card_values_display = []
gaknemu = False
salah_bang = False
is_player_turn = True
button_color = (0, 128, 255)
button_hover_color = (255, 165, 0)
button_text_color = (255, 255, 255)
button_width =200
button_height =50
GREEN = (0, 255, 0)
RED = (255, 0, 0)
card_image = []
resize_width = 560
resize_height = 420
button_stand = pygame.Rect(800, 325, 150, 50)
button_reset = pygame.Rect(800, 425, 150, 50)
button_hit = pygame.Rect(800, 525, 150, 50)
button_quit = pygame.Rect(0, 0, 150, 50)

gameover = False
processed_boxes = []
predicted_cards = []


button_reset1 = pygame.Rect(800, 325, 150, 50)
button_hit1 = pygame.Rect(800, 425, 150, 50)
button_quit1 = pygame.Rect(20, 20, 150, 50)
play_button = pygame.Rect((window_width // 2 - button_width // 2), 270, button_width, button_height)
tutorial_button = pygame.Rect((window_width // 2 - button_width // 2), 370, button_width, button_height)
exit_button = pygame.Rect((window_width // 2 - button_width // 2), 470, button_width, button_height)

exit_button1 = pygame.Rect(225, 470, button_width, button_height)
Rematch_button = pygame.Rect(575, 470, button_width, button_height)
total_deck_cards=52

model = load_model("card_classifier_model.keras")

class_labels = [
    "AS keriting", "AS wajik", "AS hati", "AS sekop",
    "delapan keriting", "delapan wajik", "delapan hati", "delapan sekop", 
    "lima keriting", "lima wajik", "lima hati", "lima sekop",
    "empat keriting", "empat wajik", "empat hati", "empat sekop",
    "jack keriting", "jack wajik", "jack hati", "jack sekop",
    "joker",
    "king keriting", "king wajik", "king hati", "king sekop",
    "sembilan keriting", "sembilan wajik", "sembilan hati", "sembilan sekop",
    "queen keriting", "queen wajik", "queen hati", "queen sekop",
    "tujuh keriting", "tujuh wajik", "tujuh hati", "tujuh sekop",
    "enam keriting", "enam wajik", "enam hati", "enam sekop",
    "sepuluh keriting", "sepuluh wajik", "sepuluh hati", "sepuluh sekop",
    "tiga keriting", "tiga wajik", "tiga hati", "tiga sekop",
    "dua keriting", "dua wajik", "dua hati", "dua sekop"
]

card_values1 = {
    "AS": "AS", "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6, "tujuh": 7,
    "delapan": 8, "sembilan": 9, "sepuluh": 10, "jack": "J", "queen": "Q", "king": "K"
}
card_values = {
    "AS": 11, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6, "tujuh": 7,
    "delapan": 8, "sembilan": 9, "sepuluh": 10, "jack": 10, "queen": 10, "king": 10
}

def load_card_image(card_name):
    card_image_path = os.path.join(f"cards/{card_name}.png")  
    try:
        card_image = pygame.image.load(card_image_path)
        card_image = pygame.transform.scale(card_image, (100, 150))  
        return card_image
    except pygame.error as e:
        print(f"Error memuat gambar {card_image_path}: {e}")
        return None
def hitung_total(deck, is_dealer=False, pemain_stand=False):
    total = 0
    as_count = 0    
   
    if is_dealer and not pemain_stand:
        card_name, value1, card_image = deck[0] 
        total += value1
        if "as" in card_name.lower():
            as_count += 1
    else:
        for card in deck:
            if len(card) == 3:
                card_name, value1, card_image = card  
            else:
                card_name, value1 = card  
            total += value1
            if "as" in card_name.lower():
                as_count += 1
    
    
    while total > 21 and as_count > 0:
        total -= 10  
        as_count -= 1
    
    return total

def display_card_image(card_image, x, y):
    if card_image is not None:
        screen.blit(card_image, (x, y))

def bobot_kartu(card_name):
    for key in card_values1:
        if key.lower() in card_name.lower():
            return card_values1[key]
    return 0
def bobot_kartu_BJ(card_name):
    for key in card_values:
        if key.lower() in card_name.lower():
            return card_values[key]
    return 0

def random_kartu(deck_dealer, deck_pemain):
    card_name = random.choice(class_labels)
    
    while "joker" in card_name.lower() or any(card[0] == card_name for card in deck_dealer) or any(card[0] == card_name for card in deck_pemain):
        card_name = random.choice(class_labels)
    
    value = bobot_kartu(card_name)
    card_image = load_card_image(card_name)  
    return card_name, value, card_image
def random_kartu_BJ(deck_dealer, deck_pemain):
    card_name = random.choice(class_labels)
    
    while "joker" in card_name.lower() or any(card[0] == card_name for card in deck_dealer) or any(card[0] == card_name for card in deck_pemain):
        card_name = random.choice(class_labels)
    
    value1 = bobot_kartu_BJ(card_name)
    card_image = load_card_image(card_name)  
    return card_name, value1, card_image

def preprocess_kartu(warped_card):
    expected_height, expected_width = model.input_shape[1:3]
    resized_card = cv.resize(warped_card, (expected_width, expected_height))
    img_array = img_to_array(resized_card) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def prediksi_kartu(model, warped_card):
    processed_card = preprocess_kartu(warped_card)
    prediction = model.predict(processed_card)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    if class_index < len(class_labels):
        card_name = class_labels[class_index]  
    else: 
        "ndak tau"
    return card_name, confidence

def draw_circle(image, points):
    for point in points:
        cv.circle(image, (int(point[0]), int(point[1])), radius=10, color=(255, 0, 0), thickness=2)
        
def draw_button(x, y, width, height, text, color, radius=20):
    pygame.draw.rect(screen, color, (x, y, width, height), border_radius=radius)  
    font = pygame.font.SysFont(None, 40)
    text_surface = font.render(text, True, button_text_color)
    screen.blit(text_surface, (x + (width - text_surface.get_width()) // 2, y + (height - text_surface.get_height()) // 2))
  
def draw_text(text, x, y, font_name="Arial", font_size=40, color=(255, 255, 255)):
    font = pygame.font.SysFont(font_name, font_size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))
def hitung_pair(deck):
    card_groups = {}
    score = 0

    for card in deck:
        card_name, value, card_image = card
        
        if value not in card_groups:
            card_groups[value] = 0
        card_groups[value] += 1

    
    for value, count in card_groups.items():
        if count >= 2:
            score += count
    
    return score

def draw_card_value_buttons(screen):
    global button_rects, card_values_display, is_player_turn, deck_pemain,awal_game  
    
    if is_player_turn:  
        if len(deck_pemain) >=4 :  
            card_values_display = set()  
            button_rects.clear()  

            
            for card in deck_pemain:
                if len(card) == 3:  
                    card_name, value, card_image = card
                elif len(card) == 2:  
                    card_name, value = card
                else:
                    continue  
                
                
                card_value_display = card_values1.get(value, str(value))  
                card_values_display.add(card_value_display)  
            
            button_width = 50
            button_height = 50
            button_spacing = 10  
            total_buttons_width = len(card_values_display) * button_width + (len(card_values_display) - 1) * button_spacing
            start_x = (window_width - total_buttons_width) // 2
            start_y = 580  

            
            for i, value in enumerate(card_values_display):
                button_text = value  
                button_rect = pygame.Rect(start_x + i * (button_width + button_spacing), start_y, button_width, button_height)
                
                
                pygame.draw.rect(screen, (255, 255, 255), button_rect)
                font = pygame.font.SysFont(None, 30)
                text_surface = font.render(button_text, True, (0, 0, 0))  
                screen.blit(text_surface, (button_rect.x + 17, button_rect.y + 17))  
                button_rects.append(button_rect)
                    

        else:
            font = pygame.font.SysFont(None, 36)
            cards_needed = 4 - len(deck_pemain)  
            message = f"Jumlah kartu pemain kurang, {cards_needed} kartu lagi"
            text_surface = font.render(message, True, (255, 0, 0))  
            text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 4))  
            screen.blit(text_surface, text_rect)
            pygame.display.update()
            
    else:
        awal_game = False
        pass

def bot_choose_card(deck_bot):
    card_counts = {}

    for card in deck_bot:
        value = card[1]
        if value in card_counts:
            card_counts[value] += 1
        else:
            card_counts[value] = 1

    possible_values = [value for value, count in card_counts.items() if count > 1]

    if possible_values:
        chosen_value = random.choice(possible_values)
        print(f"Bot memilih untuk meminta kartu dengan nilai {chosen_value}.")
        return chosen_value
    else:
        if deck_bot:  
            chosen_card = random.choice(deck_bot)
            print(f"Bot memilih kartu acak dengan nilai {chosen_card[1]} dan menambahkannya ke deck bot.")
            return chosen_card[1]
        else:
            print("Deck bot kosong, tidak ada kartu yang bisa dipilih.")
            return None 
def konfirmasi():
    global is_player_turn, deck_pemain, deck_bot, confirm_button, salah_bang
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if is_player_turn and salah_bang:  
        
        confirm_button = pygame.Rect(400, 500, 200, 50)
        pygame.draw.rect(screen, (0, 255, 0), confirm_button)  
        font = pygame.font.Font(None, 36)
        text_surface = font.render("Konfirmasi", True, (0, 0, 0))
        screen.blit(text_surface, (confirm_button.x + 10, confirm_button.y + 10))
        pygame.display.update()

       
           
def handle_button_click(mouse_x, mouse_y, button_rects, card_values_display, predicted_cards):
    global is_player_turn, deck_pemain, deck_bot, confirm_button, salah_bang,card_image

    card_values_display = list(card_values_display)
    
    if is_player_turn: 
        for i, button_rect in enumerate(button_rects):
            if button_rect.collidepoint(mouse_x, mouse_y):  
                value = card_values_display[i] 

                if value.isdigit():
                    value_to_check = int(value)  
                else:
                    value_to_check = value  

                cards_to_move = [card for card in deck_bot if card[1] == value_to_check]  

                if cards_to_move:
                    print(f"Bot memiliki kartu {value}.")
                    
                    for card in cards_to_move:
                        if card not in deck_pemain:  
                            deck_pemain.append(card)
                            draw_cards_and_buttons(deck_pemain, deck_bot, screen,card_image)
                            deck_bot.remove(card)

                    pygame.display.update()

                    print(f"Kartu dengan value {value} berhasil dipindahkan ke deck pemain.")
                    #is_player_turn = False
                else:
                    
                    print(f"Kartu dengan value {value} tidak ada di deck bot.")
                    salah_bang = True  
                    break  

    
            
    if not is_player_turn:  
        print("Giliran bot dimulai.")
        chosen_value = bot_choose_card(deck_bot)

        if chosen_value:
            font = pygame.font.Font(None, 36)
            message = f"Bot meminta kartu dengan nilai: {chosen_value}"
            text_surface = font.render(message, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 4))
            screen.blit(text_surface, text_rect)
            pygame.display.update()

            pygame.time.delay(1000)

            cards_to_move = [card for card in deck_pemain if card[1] == chosen_value]

            if cards_to_move:
                print(f"Pemain memberikan kartu {chosen_value} ke bot.")
                for card in cards_to_move:
                    deck_bot.append(card)
                    deck_pemain.remove(card)
                
                pygame.display.update()
                message = f"Kartu dengan nilai {chosen_value} dipindahkan ke deck bot."
                text_surface = font.render(message, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 4 + 30))
                screen.blit(text_surface, text_rect)
                pygame.display.update()
                pygame.time.delay(1000)
                
            else:
                
                message = f"Bot mengambil kartu acak."
                text_surface = font.render(message, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 4 + 30))
                screen.blit(text_surface, text_rect)
                pygame.display.update()
                card_name, value, card_image = random_kartu(deck_bot, deck_pemain)
                deck_bot.append((card_name, value, card_image))
                pygame.display.update()
                pygame.time.delay(1000)
                print("Kartu acak telah ditambahkan ke deck bot.")
                is_player_turn = True

        else:
            print("Giliran bot berakhir.")

    return is_player_turn


def draw_cards_and_buttons(deck_pemain, deck_bot, screen,card_back_image):
    num_dealer_cards = len(deck_bot)
    num_pemain_cards = len(deck_pemain)
    card_spacing = 20

    total_pemain_cards_width = num_pemain_cards * card_spacing - card_spacing
    total_cards_width = num_dealer_cards * card_spacing - card_spacing  

    
    center_x_pemain = (window_width - total_pemain_cards_width) // 2
    center_x = (window_width - total_cards_width) // 2 
 
    for i, (card_name, value, card_image) in enumerate(deck_bot):
        card_x = center_x + i * card_spacing
        if card_image is not None:
            screen.blit(card_back_image, (card_x - 50, 30)) 

    
    for i, card in enumerate(deck_pemain):
        if len(card) == 3:
            card_name, value, card_image = card
        else:
            card_name, value = card
            card_image = None

        card_x_pemain = center_x_pemain + i * card_spacing
        if card_image is not None:
            screen.blit(card_image, (card_x_pemain - 50, 630)) 

    
    draw_card_value_buttons(screen) 

def kalo_empat(deck):
    card_groups = {}

    for card in deck:
        card_name, value, card_image = card
        suit = card_name.split()[-1] 
        
        if value not in card_groups:
            card_groups[value] = set()  
        
        card_groups[value].add(suit)  
    
    for value, suits in card_groups.items():
        if len(suits) == 4:
            return True

    return False

def main_game():
    cam = cv.VideoCapture(0)

    if not cam.isOpened():
        print("tidak dapat mengakses kamera")
        exit()

    deck_pemain = []
    deck_dealer = []
    total_pemain = 0
    total_dealer = 0
    gameover = False
    pemain_stand = False
    #frame_counter = 0
    processed_boxes = []
    predicted_cards = []  
    hasil_permainan = ""
    warna_hasil = (255, 255, 255)

    lower = np.array([40, 40, 40])
    upper = np.array([80, 255, 255])




    for _ in range(2):
        card_name, value1, card_image = random_kartu_BJ(deck_dealer, deck_pemain)
        deck_dealer.append((card_name, value1, card_image))  
        
    card_back_image = pygame.image.load("Back Red.png")
    card_back_image = pygame.transform.scale(card_back_image, (100, 150))

    while True:
        ret, image = cam.read()
        if not ret:
            print("Error dalam mengambil frame")
            break
        
        #frame_counter += 1
        blurred = cv.GaussianBlur(image, (5, 5), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)
        mask = cv.bitwise_not(mask)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.GaussianBlur(mask, (5, 5), 0)
        
        #cv.imshow('hasil mask', mask)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv.contourArea(contour)
            if area < 500:
                continue

            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                pts = np.array([point[0] for point in approx], dtype="float32")
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
                height = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))
                dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
                M = cv.getPerspectiveTransform(rect, dst)
                warped = cv.warpPerspective(image, M, (width, height))

                if width > height:
                    warped = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)

                box_key = (x, y, w, h)
                already_processed = any(np.allclose(box_key, processed, atol=20) for processed in processed_boxes)

                if not already_processed: #and frame_counter % 10 == 0:
                    try:
                        card_name, confidence = prediksi_kartu(model, warped)
                        if confidence > 0.2:
                            value1 = bobot_kartu_BJ(card_name)
                            deck_pemain.append((card_name, value1, load_card_image(card_name)))
                            processed_boxes.append(box_key)
                            predicted_cards.append((card_name, confidence))  
                            cv.putText(image, f"{card_name} ({confidence:.2f})", (x, y - 10), 
                                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Error prediksi: {e}")

                draw_circle(image, rect)
            
            total_pemain = hitung_total(deck_pemain)
            
            

            
        if not gameover:
            if pemain_stand:
                total_dealer = hitung_total(deck_dealer)
                while total_dealer < 17:
                    card_name, value1, card_image = random_kartu_BJ(deck_dealer, deck_pemain)
                    deck_dealer.append((card_name, value1,card_image))
                    total_dealer += value1

                if total_dealer > 21 or total_dealer < total_pemain:
                    hasil_permainan = "Pemain Menang!"
                    warna_hasil = (0, 255, 0)
                elif total_dealer > total_pemain:
                    hasil_permainan = "Dealer Menang!"
                    warna_hasil = (255, 0, 0)
                else:
                    hasil_permainan = "Seri!"
                    warna_hasil = (255, 255, 0)
                gameover = True
            elif total_pemain > 21:
                hasil_permainan = "Pemain BUST! Dealer Menang!"
                warna_hasil = (255, 0, 0)
                gameover = True
            elif total_pemain == 21:
                hasil_permainan = "Blackjack! Pemain Menang!"
                warna_hasil = (0, 255, 0)
                gameover = True
            else:
                total_dealer = deck_dealer[0][1]

        score_window = np.zeros((400, 500, 3), dtype=np.uint8)  
        cv.putText(score_window, f"Pemain: {total_pemain}", (10, 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        y_offset = 100  
        for card in deck_pemain:
            cv.putText(score_window, card[0], (10, y_offset), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 20  

        if not pemain_stand:
            if len(deck_dealer[0]) == 3:
                dealer_card_name, value1, card_image = deck_dealer[0]  
            else:
                dealer_card_name, value1 = deck_dealer[0]  
                card_image = None 

            cv.putText(score_window, f"Dealer: {dealer_card_name} + ?", (10, y_offset+30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            dealer_cards = ", ".join([card[0] for card in deck_dealer])
            y_offset += 30  
            cv.putText(score_window, f"Dealer: {total_dealer}", (10, y_offset), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            y_offset += 30  
            for card in deck_dealer:
                cv.putText(score_window, card[0], (10, y_offset), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 20  

        

        for box, card in zip(processed_boxes, predicted_cards):
            x, y, w, h = box
            #cv.putText(image, f"{card[0]} ({card[1]:.2f})", (x, y - 10), 
                      # cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                     
                        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (resize_width, resize_height))
        frame_surface = pygame.surfarray.make_surface(image)
        frame_surface = pygame.transform.rotate(frame_surface, -270)
        frame_surface = pygame.transform.flip(frame_surface, False, True)  
        screen.blit(background_image, (0, 0))
        
        screen.blit(frame_surface, ((window_width - resize_width) // 2, (window_height - resize_height) // 2))
        if gameover:
            draw_text(f"Pemain: {total_pemain}", 50, 650, font_size=40, color=GREEN)
            draw_text(f"Dealer: {total_dealer}", 50, 100, font_size=40, color=RED)
            draw_text(hasil_permainan, 10, 400, font_size=40, color=warna_hasil)
        if not gameover :
            draw_text(f"Pemain: {total_pemain}", 50, 650, font_size=40, color=GREEN)
            draw_text(f"Dealer: {total_dealer}", 50, 100, font_size=40,color=RED)
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if button_stand.collidepoint(mouse_x, mouse_y):
            draw_button(800, 325, 150, 50, "STAND", button_hover_color)
        else:
            draw_button(800, 325, 150, 50, "STAND", button_color)        
        if button_reset.collidepoint(mouse_x, mouse_y):
            draw_button(800, 425, 150, 50, "RESET", button_hover_color)
        else:
            draw_button(800, 425, 150, 50, "RESET", button_color)
        if button_hit.collidepoint(mouse_x, mouse_y):
            draw_button(800, 525, 150, 50, "HIT", button_hover_color)
        else:
            draw_button(800, 525, 150, 50, "HIT", button_color)
        if button_quit.collidepoint(mouse_x, mouse_y):
            draw_button(0, 0, 150, 50, "QUIT", button_hover_color)
        else:
            draw_button(0, 0, 150, 50, "QUIT", button_color)

        num_dealer_cards = len(deck_dealer)
        num_pemain_cards = len(deck_pemain)
        
        card_width = 100  
        card_spacing = 120  
        
        
        total_pemain_cards_width = num_pemain_cards * card_spacing - card_spacing
        total_cards_width = num_dealer_cards * card_spacing - card_spacing  
        
        
        center_x_pemain = (window_width - total_pemain_cards_width) // 2
        center_x = (window_width - total_cards_width) // 2
        for i, (card_name, value1, card_image) in enumerate(deck_dealer):
            card_x = center_x + i * card_spacing
            if not pemain_stand and i == 1:  
                screen.blit(card_back_image, (card_x-50, 30))  
            else:  
                if card_image is not None:
                    screen.blit(card_image, (card_x-50, 30))  

        for i, card in enumerate(deck_pemain):
            # Unpack kartu sesuai dengan jumlah elemen
            if len(card) == 3:
                card_name, value1, card_image = card  # Kartu dengan gambar
            else:
                card_name, value1 = card  # Kartu tanpa gambar
                card_image = None  # Set gambar menjadi None jika tidak ada gambar
        
            # Hitung posisi x untuk memusatkan kartu pemain
            card_x_pemain = center_x_pemain + i * card_spacing
        
            # Tampilkan gambar kartu pemain jika ada gambar
            if card_image is not None:  # Pastikan gambar ada
                screen.blit(card_image, (card_x_pemain-50, 630))
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos

                if button_stand.collidepoint(mouse_x, mouse_y) and not gameover:
                    pemain_stand = True  
                    
                if button_quit.collidepoint(mouse_x, mouse_y) :
                    menu()
                if button_hit.collidepoint(mouse_x, mouse_y):
                     processed_boxes = []
                     predicted_cards = [] 
                     total_pemain = 0
                     deck_pemain = []
                     gameover = False
                if button_reset.collidepoint(mouse_x, mouse_y):
                    deck_pemain = []
                    deck_dealer = []
                    total_pemain = 0
                    total_dealer = 0
                    gameover = False
                    pemain_stand = False
                    processed_boxes = []
                    predicted_cards = []  
                    
                    for _ in range(2):
                        card_name, value, card_image = random_kartu_BJ(deck_dealer, deck_pemain)
                        deck_dealer.append((card_name, value, card_image))
                            
        card_back_image = pygame.image.load("Back Red.png")
        card_back_image = pygame.transform.scale(card_back_image, (100, 150))
        pygame.display.update()

        #cv.imshow('Frame', image)
        #cv.imshow("Skor blackjack ", score_window) 

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        #elif key == ord('s') and not pemain_stand and not gameover:
            #pemain_stand = True
        #elif key == ord('r'):
def GO_FISH():
    cam = cv.VideoCapture(1)
    if not cam.isOpened():
        print("Tidak dapat mengakses kamera")
        exit()
    global deck_bot, deck_pemain, is_player_turn,confirm_button,salah_bang,awal_game,gameover,processed_boxes ,predicted_cards,is_player_turn
    confirm_button = pygame.Rect(400, 500, 200, 50)
    deck_pemain = []
    deck_bot = []
    gameover = False
    awal_game = True
    is_player_turn = True
    processed_boxes = []
    predicted_cards = []
    salah_bang = False  
    hasil_permainan = ""
    
    warna_hasil = (255, 255, 255)

    lower = np.array([40, 40, 40])
    upper = np.array([80, 255, 255])

    for _ in range(4):
        card_name, value, card_image = random_kartu(deck_bot, deck_pemain)
        deck_bot.append((card_name, value, card_image))  
    
    card_back_image = pygame.image.load("Back Red.png")
    card_back_image = pygame.transform.scale(card_back_image, (100, 150))

    while True:
        ret, image = cam.read()
        if not ret:
            print("Error dalam mengambil frame")
            break
        blurred = cv.GaussianBlur(image, (5, 5), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)
        mask = cv.bitwise_not(mask)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.GaussianBlur(mask, (5, 5), 0)
        
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv.contourArea(contour)
            if area < 500:
                continue

            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                pts = np.array([point[0] for point in approx], dtype="float32")
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
                height = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))
                dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
                M = cv.getPerspectiveTransform(rect, dst)
                warped = cv.warpPerspective(image, M, (width, height))

                if width > height:
                    warped = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)

                box_key = (x, y, w, h)
                already_processed = any(np.allclose(box_key, processed, atol=20) for processed in processed_boxes)

                if not already_processed:
                    try:
                        
                        card_name, confidence = prediksi_kartu(model, warped)
                        
                        if confidence > 0.2:
                            
                            kartu_ada = any(card[0] == card_name for card in deck_bot)
                            kartu_ada_pemain = any(card[0] == card_name for card in deck_pemain)
                            if kartu_ada:
                               
                                print(f"PERINGATAN: Kartu {card_name} sudah ada di deck bot dan tidak dapat ditambahkan ke deck pemain.")
                                
                                cv.putText(image, f"Peringatan: {card_name} ada di deck bot", (x, y - 20), 
                                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            elif kartu_ada_pemain:
                                
                                print(f"PERINGATAN: Kartu {card_name} sudah ada di deck pemain dan tidak dapat ditambahkan.")
                               
                                cv.putText(image, f"Peringatan: {card_name} ada di deck pemain", (x, y - 20),
                                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:
                                value = bobot_kartu(card_name)
                                deck_pemain.append((card_name, value, load_card_image(card_name)))
                                processed_boxes.append(box_key)
                                predicted_cards.append((card_name, confidence))
                                

                                cv.putText(image, f"{card_name} ({confidence:.2f})", (x, y - 10), 
                                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                draw_circle(image, rect)
                    
                    except Exception as e:
                        print(f"Error prediksi: {e}")
                    
                    
                    draw_circle(image, rect)

        if kalo_empat(deck_pemain):   
            warna_hasil = (0, 255, 0)
            pemenang("PEMAIN",warna_hasil)
            gameover = True
        elif kalo_empat(deck_bot):
            warna_hasil = (255, 0, 0)
            pemenang("BOT",warna_hasil)
            gameover = True
          
        if gameover:
            cv.putText(image, hasil_permainan, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, warna_hasil, 2)
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (resize_width, resize_height))
        frame_surface = pygame.surfarray.make_surface(image)
        frame_surface = pygame.transform.rotate(frame_surface, -270)
        frame_surface = pygame.transform.flip(frame_surface, False, True)

        
        screen.blit(background_image, (0, 0))

        screen.blit(frame_surface, ((window_width - resize_width) // 2, (window_height - resize_height) // 2))

        mouse_x, mouse_y = pygame.mouse.get_pos()
        if button_reset1.collidepoint(mouse_x, mouse_y):
            draw_button(800, 325, 150, 50, "RESET", button_hover_color)
        else:
            draw_button(800, 325, 150, 50, "RESET", button_color)
        if button_hit1.collidepoint(mouse_x, mouse_y):
            draw_button(800, 425, 150, 50, "SCAN", button_hover_color)
        else:
            draw_button(800, 425, 150, 50, "SCAN", button_color)
        if button_quit1.collidepoint(mouse_x, mouse_y):
            draw_button(20, 20, 150, 50, "QUIT", button_hover_color)
        else:
            draw_button(20, 20, 150, 50, "QUIT", button_color)

        
        num_dealer_cards = len(deck_bot)
        num_pemain_cards = len(deck_pemain)

        card_spacing = 20  
        total_pemain_cards_width = num_pemain_cards * card_spacing - card_spacing
        total_cards_width = num_dealer_cards * card_spacing - card_spacing  

        center_x_pemain = (window_width - total_pemain_cards_width) // 2
        center_x = (window_width - total_cards_width) // 2 
        
        
        for i, (card_name, value, card_image) in enumerate(deck_bot):
            card_x = center_x + i * card_spacing
            if card_image is not None:
                screen.blit(card_image, (card_x - 50, 30))  \

        for i, card in enumerate(deck_pemain):
            if len(card) == 3:
                card_name, value, card_image = card  
            else:
                card_name, value = card  
                card_image = None  

            card_x_pemain = center_x_pemain + i * card_spacing  
            if card_image is not None:  
                screen.blit(card_image, (card_x_pemain - 50, 630))  
        
        remaining_deck = total_deck_cards - len(deck_pemain) - len(deck_bot)
        y_pos = 50
        for i in range(remaining_deck):
            if y_pos + 150 < window_height:
                screen.blit(card_back_image, (50, y_pos+50))  
                y_pos += 10
        score_pemain = hitung_pair(deck_pemain)
        score_bot = hitung_pair(deck_bot)
        if remaining_deck == 0:
            if score_pemain > score_bot:   
                warna_hasil = (0, 255, 0)
                pemenang("PEMAIN",warna_hasil)
                gameover = True
            elif score_pemain < score_bot:
                warna_hasil = (255, 0, 0)
                pemenang("BOT",warna_hasil)
                gameover = True
            elif score_pemain == score_bot:
                warna_hasil = (0, 0, 255)
                pemenang("TIDAK ADA",warna_hasil)
                gameover = True
        draw_cards_and_buttons(deck_pemain, deck_bot, screen,card_back_image)

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                if button_quit1.collidepoint(mouse_x, mouse_y):
                    menu()
                if button_hit1.collidepoint(mouse_x, mouse_y):
                    processed_boxes = []
                    predicted_cards = [] 
                if button_reset1.collidepoint(mouse_x, mouse_y):
                    deck_pemain = []
                    deck_bot = []
                    gameover = False
                    awal_game = True
                    processed_boxes = []
                    predicted_cards = []
                    salah_bang = False
                    for _ in range(4):
                        card_name, value, card_image = random_kartu(deck_bot, deck_pemain)
                        deck_bot.append((card_name, value, card_image))
                if confirm_button.collidepoint(mouse_x, mouse_y):
                    is_player_turn = False  
                    confirm_button = pygame.Rect(0, 0, 0, 0)  
                    salah_bang = False  
                    print("Giliran sekarang berpindah ke bot setelah konfirmasi.") 
                handle_button_click(mouse_x, mouse_y, button_rects, card_values_display, predicted_cards)
                    
            
        konfirmasi()
        pygame.display.update()
def pemenang(pemenang,warna): 
    global deck_bot, deck_pemain, is_player_turn,confirm_button,salah_bang,awal_game,gameover,processed_boxes ,predicted_cards 
    while True:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        background_image = pygame.image.load("BG_MENU.jpg")
        background_image = pygame.transform.scale(background_image, (window_width, window_height))
        screen.blit(background_image, (0, 0))

        font = pygame.font.SysFont(None, 72)  
        winner_text = f"{pemenang} MENANG!" 
        text_surface = font.render(winner_text, True, warna)  # 
        text_rect = text_surface.get_rect(center=(window_width // 2, window_height // 2 - 30))
        screen.blit(text_surface, text_rect)

        if Rematch_button.collidepoint(mouse_x, mouse_y):
            draw_button(Rematch_button.x, Rematch_button.y, button_width, button_height, "Rematch", button_hover_color)
        else:
            draw_button(Rematch_button.x,Rematch_button.y, button_width, button_height, "Rematch", button_color)

        if exit_button1.collidepoint(mouse_x, mouse_y):
            draw_button(exit_button1.x, exit_button1.y, button_width, button_height, "Exit", button_hover_color)
        else:
            draw_button(exit_button1.x, exit_button1.y, button_width, button_height, "Exit", button_color)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Rematch_button.collidepoint(event.pos):
                    deck_pemain = []
                    deck_bot = []
                    gameover = False
                    awal_game = True
                    processed_boxes = []
                    predicted_cards = []
                    salah_bang = False
                    menu() 
                if exit_button1.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
        pygame.display.update()

def menu():
    while True:
        background_image = pygame.image.load("BG_MENU.jpg")
        background_image = pygame.transform.scale(background_image, (window_width, window_height))
        screen.blit(background_image, (0, 0))

        mouse_x, mouse_y = pygame.mouse.get_pos()

        if play_button.collidepoint(mouse_x, mouse_y):
            draw_button(play_button.x, play_button.y, button_width, button_height, "BlackJack!", button_hover_color)
        else:
            draw_button(play_button.x, play_button.y, button_width, button_height, "BlackJack", button_color)

        if tutorial_button.collidepoint(mouse_x, mouse_y):
            draw_button(tutorial_button.x, tutorial_button.y, button_width, button_height, "Go Fish", button_hover_color)
        else:
            draw_button(tutorial_button.x, tutorial_button.y, button_width, button_height, "Go Fish", button_color)

        if exit_button.collidepoint(mouse_x, mouse_y):
            draw_button(exit_button.x, exit_button.y, button_width, button_height, "EXIT", button_hover_color)
        else:
            draw_button(exit_button.x, exit_button.y, button_width, button_height, "EXIT", button_color)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if play_button.collidepoint(event.pos):
                    main_game()  
                if tutorial_button.collidepoint(event.pos):
                    GO_FISH()
                if exit_button.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()
#GO_FISH()    
menu()    
cv.destroyAllWindows()
pygame.quit()