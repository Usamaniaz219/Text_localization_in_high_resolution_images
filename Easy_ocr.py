import easyocr
import cv2

# def extract_character_coordinates(result):
#     bounding_boxes = []
#     for bbox, text, confidence in result:
#         try:
#             [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
#         except ValueError:
#             continue
#         bounding_boxes.append((int(x1), int(y1), int(x3), int(y3)))
    
#     return bounding_boxes

reader = easyocr.Reader(['en'], gpu=True)
image_path = '/home/usama/diffBIR_results/LUV__low_res_Meanshift_Bandwidth_8/data/demo133.jpg'
image = cv2.imread(image_path)
result = reader.readtext(image, text_threshold=0.6, low_text=0.5, width_ths=0.8,workers = 6)
# print(result)
bounding_boxes = [bbox for bbox, _, _ in result]
# print("Bounding Boxes",bounding_boxes)
for bbox in bounding_boxes:
    x1, y1 = bbox[0]
    #print("x1",x1)
    x3, y3 = bbox[2] 

    print(bbox,",")
    # print(",")
    # print("X1,x3",x1,x3)
    # print("y1, y3", y1, y3)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x3), int(y3)), (0, 0, 255), 2)
    cv2.imwrite('Easy_ocr_output_@@.jpg', image)

# result = reader.readtext(image_path, text_threshold=0.06, low_text=0.01, width_ths=0.8)

# bounding_boxes = extract_character_coordinates(result)

# for x1, y1, x3, y3 in bounding_boxes:
#     cv2.rectangle(image, (x1, y1), (x3, y3), (0, 0, 255), 2)
#     cv2.putText(image, "", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# cv2.imwrite('Easy_ocr_output_default.jpg', image)
# print("Bounding Boxes", bounding_boxes)



# import easyocr
# import cv2
# import numpy as np

# def extract_character_contours(result):
#     character_contours = []
#     for bbox, text, confidence in result:
#         try:
#             [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
#         except ValueError:
#             continue
#         character_contours.append([(int(x), int(y)) for x, y in bbox])
    
#     return character_contours

# reader = easyocr.Reader(['en'], gpu=False)
# image_path = '/home/usama/diffBIR_results/LUV__low_res_Meanshift_Bandwidth_8/New_results/demo133/demo133_1.jpg'
# image = cv2.imread(image_path)

# result = reader.readtext(image_path, text_threshold=0.06, low_text=0.01, width_ths=0.8)

# character_contours = extract_character_contours(result)

# for contour in character_contours:
#     np_contour = np.array(contour, dtype=np.int32)
#     cv2.drawContours(image, [np_contour], 0, (0, 0, 255), 2)

# cv2.imwrite('Easy_ocr_output_default.jpg', image)
# print("Character Contours", character_contours)



# import easyocr
# import cv2
# import numpy as np

# def extract_character_contours(result):
#     character_contours = []
#     for bbox, text, confidence in result:
#         try:
#             [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
#         except ValueError:
#             continue
#         character_contours.append([(int(x), int(y)) for x, y in bbox])
    
#     return character_contours

# reader = easyocr.Reader(['en','es'], gpu=False)
# image_path = '/home/usama/diffBIR_results/LUV__low_res_Meanshift_Bandwidth_8/New_results/demo133/demo133_4.jpg'
# image = cv2.imread(image_path)
# result = reader.readtext(image_path,add_margin=0.05,text_threshold=0.5,low_text=0.01,mag_ratio=0.55,height_ths =0.3)
# # result = reader.readtext(image_path, text_threshold=0.06, low_text=0.01, width_ths=0.8)

# character_contours = extract_character_contours(result)

# for contour in character_contours:
#     np_contour = np.array(contour, dtype=np.int32)
#     cv2.fillPoly(image, [np_contour], (0, 0, 255))

# cv2.imwrite('Easy_ocr_output_default.jpg', image)
# print("Character Contours", character_contours)




# # Import the required libraries
# import easyocr
# import cv2
# import numpy as np

# # Define a function to extract the coordinates of each character from the result
# def extract_character_coordinates(result):
#     character_coordinates = []
#     for bbox, text, confidence in result:
#         print("text",text)
#         try:
#             # Get the coordinates of the bounding box of the text
#             [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
#         except ValueError:
#             continue
#         # Get the width and height of the bounding box
#         width = x2 - x1
#         height = y3 - y2
#         # Get the number of characters in the text
#         num_chars = len(text)
#         print("Number of characters:", num_chars)
#         # Calculate the average width of each character
#         char_width = width / num_chars
#         # Loop through the characters in the text
#         for i, char in enumerate(text):
#             # Calculate the coordinates of the character
#             char_x1 = int(x1 + i * char_width)
#             char_x2 = int(x1 + (i + 1) * char_width)
#             char_y1 = int(y1)
#             char_y2 = int(y3)
#             # Append the coordinates to the list
#             character_coordinates.append([(char_x1, char_y1), (char_x2, char_y1), (char_x2, char_y2), (char_x1, char_y2)])

#     return character_coordinates

# # Initialize the easyocr reader
# reader = easyocr.Reader(['en','es'], gpu=False)
# # Read the image path
# image_path = '/home/usama/diffBIR_results/LUV__low_res_Meanshift_Bandwidth_8/New_results/demo133/demo133_2.jpg'
# # Read the image
# image = cv2.imread(image_path)
# # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
# # image =cv2.GaussianBlur(image,(5,5),0)
# # image = cv2.medianBlur(image, 5)
# # Get the result from the reader
# result = reader.readtext(image_path, add_margin=0.05, text_threshold=0.5, low_text=0.01, mag_ratio=0.2)
# # Extract the character coordinates from the result
# character_coordinates = extract_character_coordinates(result)

# # Loop through the character coordinates
# for coord in character_coordinates:
#     # Convert the coordinates to a numpy array
#     np_coord = np.array(coord, dtype=np.int32)
#     # Fill the character with a red color on the image
#     # cv2.fillPoly(image, [np_coord], (0, 0, 255))
#     cv2.drawContours(image, [np_coord], 0, (0, 0, 255), 1)

# # Save the image
# cv2.imwrite('Easy_ocr_output_default.jpg', image)
# # Print the character coordinates
# print("Character Coordinates", character_coordinates)



# # Import the required libraries
# import easyocr
# import cv2
# import numpy as np

# # Define a function to extract the edges of each character from the result
# def extract_character_edges(result):
#     character_edges = []
#     for bbox, text, confidence in result:
#         print("text", text)
        
#         try:
#             # Get the coordinates of the bounding box of the text
#             [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
#         except ValueError:
#             continue
#         # Get the width and height of the bounding box
#         width = x2 - x1
#         height = y3 - y2
#         # Get the number of characters in the text
#         num_chars = len(text)
#         print("Number of characters:", num_chars)
#         # Calculate the average width of each character
#         char_width = width / num_chars
#         # Loop through the characters in the text
#         for i, char in enumerate(text):
#             print("char", char)
#             # Calculate the coordinates of the character
#             char_x1 = int(x1 + i * char_width)
#             char_x2 = int(x1 + (i + 1) * char_width)
#             char_y1 = int(y1)
#             char_y2 = int(y3)
#             # Append the coordinates to the list
#             character_edges.append([(char_x1, char_y1), (char_x2, char_y1), (char_x2, char_y2), (char_x1, char_y2)])

#     return character_edges

# # Initialize the easyocr reader
# reader = easyocr.Reader(['en', 'es'], gpu=True)
# # Read the image path
# image_path = '/home/usama/diffBIR_results/LUV__low_res_Meanshift_Bandwidth_8/data/demo133.jpg'
# # Read the image
# image = cv2.imread(image_path)
# # Get the result from the reader
# # result = reader.readtext(image_path, text_threshold=0.6, low_text=0.32, mag_ratio=2,min_size =10,decoder='beamsearch',beamWidth = 100)
# result = reader.readtext(image_path,mag_ratio=2.85,text_threshold=0.7,low_text=0.001,min_size=5)
# # Extract the character edges from the result
# character_edges = extract_character_edges(result)

# # Loop through the character edges
# for edge in character_edges:
#     # Convert the edges to a numpy array
#     np_edge = np.array(edge, dtype=np.int32)
#     # Draw the edges with a red color on the image
#     cv2.polylines(image, [np_edge], isClosed=True, color=(0, 0, 255), thickness=1)

# # Save the image
# cv2.imwrite('Easy_ocr_output_default.jpg', image)
# # Print the character edges
# print("Character Edges", character_edges)




































# import easyocr
# import cv2
# reader = easyocr.Reader(['en'],gpu=False) # this needs to run only once to load the model into memory
# image_path='/home/usama/diffBIR_results/LUV__low_res_Meanshift_Bandwidth_8/New_results/demo133/demo133_1.jpg'
# image=cv2.imread(image_path)
# # result = reader.readtext(image_path,add_margin=0.05,mag_ratio=2.85,text_threshold=0.7)
# result = reader.readtext(image_path,text_threshold=0.06,low_text = 0.01,width_ths=0.8)
# #result = reader.readtext(image_path,text_threshold=0.8,rotation_info=[90,180,270,360],y_ths=0.9,mag_ratio=18,
#  #                                    canvas_size =6000,adjust_contrast=0.7,contrast_ths=0.2,min_size=5)
# # #print(result)
# # result = reader.detect(image_path)
# # result1 = reader.recognize(image_path,free_list = result)
# # print("result",result1)

# bounding_boxes=[]
# for bbox ,text, confidence in result:
#                     try:
#                         [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox

#                     except ValueError:
#                         continue
#                     bounding_boxes.append(bbox)
            
# for bbox in bounding_boxes:
# #print("BBOX:",bbox) 
#     x1, y1 = bbox[0]
#     x3, y3 = bbox[2]     
#     cv2.rectangle(image, (int(x1), int(y1)), (int(x3), int(y3)), (0, 0, 255), 2)
# # cv2.imwrite('Easy_ocr_output_default.jpg',image)  
# print("Bounding Boxes",bounding_boxes)  
# cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# cv2.imwrite('Easy_ocr_output_default.jpg',image)  


# # # # Show the tile with bounding box
# # # cv2.imshow('Tile with Bounding Box',image)
# # # cv2.waitKey(0)
# # cv2.destroyAllWindows()               

#                     #mapped_bbox =  [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
#                     #print("Mapped BBox", mapped_bbox)

# #transform_image(image_path, 'transformed_image11.jpg', 320, 320)






  