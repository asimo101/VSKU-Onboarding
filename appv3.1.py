import streamlit as st
import pandas as pd
import os
import requests
import hashlib
from PIL import Image
import imagehash
import io
from io import BytesIO

# Set temp directory
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

st.set_page_config(layout="wide")
st.title("Image Grouping Application")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

# Initialize session state
if "ordered_groups" not in st.session_state:
    st.session_state.ordered_groups = {}
if "ungrouped_images" not in st.session_state:
    st.session_state.ungrouped_images = []
if "image_info" not in st.session_state:
    st.session_state.image_info = {}
if "removed_images" not in st.session_state:
    st.session_state.removed_images = []
if "uniware_sku_codes" not in st.session_state:
    st.session_state.uniware_sku_codes = {}
if "image_level_sku_codes" not in st.session_state:
    st.session_state.image_level_sku_codes = {}
if "unassigned_images" not in st.session_state:
    st.session_state.unassigned_images = []
if "show_unassigned" not in st.session_state:
    st.session_state.show_unassigned = False
if "current_group_idx" not in st.session_state:
    st.session_state.current_group_idx = 0

if uploaded_file:
    if "grouped" not in st.session_state:
        st.session_state.grouped = False
        st.session_state.image_info = {}
        st.session_state.hashes = {}
        st.session_state.ordered_groups = {}
        st.session_state.unassigned_images = []
        st.session_state.uniware_sku_codes = {}
        st.session_state.image_level_sku_codes = {}
        st.session_state.show_unassigned = False

    if not st.session_state.grouped:
        df = pd.read_excel(uploaded_file)
        required_columns = {"Channel Name", "Channel Product Id", "Seller SKU Code", "Product name", "Channel Code", "Image URL"}
        if not required_columns.issubset(df.columns):
            st.error("Excel must contain the required columns.")
        else:
            image_info = {}
            hashes = {}

            total_images = len(df)
            progress_download = st.progress(0, text="Downloading Images...")

            for idx, (_, row) in enumerate(df.iterrows()):
                url = row['Image URL']
                seller_sku = row['Seller SKU Code']
                product_name = row['Product name']
                channel_code = row['Channel Code']
                channel_name = row['Channel Name']
                channel_product_id = row['Channel Product Id']

                #image_name = hashlib.md5(url.encode()).hexdigest() + ".jpg"
                image_name = channel_product_id + channel_code + ".jpg"
                image_path = os.path.join(temp_dir, image_name)

                if not os.path.exists(image_path):
                    try:
                        response = requests.get(url, timeout=10)
                        img = Image.open(BytesIO(response.content)).convert('RGB')
                        img.save(image_path)
                    except:
                        continue

                image_info[image_path] = (channel_name, channel_product_id, seller_sku, product_name, channel_code)
                progress_download.progress((idx + 1) / total_images, text=f"Downloaded {idx + 1}/{total_images}")

            progress_hash = st.progress(0, text="Hashing Images...")

            for idx, img_path in enumerate(image_info.keys()):
                try:
                    img = Image.open(img_path)
                    img_hash = imagehash.phash(img)
                    hashes[img_path] = img_hash
                except:
                    continue
                progress_hash.progress((idx + 1) / total_images, text=f"Hashed {idx + 1}/{total_images}")

            progress_group = st.progress(0, text="Grouping Images...")
            group_list = []
            visited = set()

            for idx, (img1, hash1) in enumerate(hashes.items()):
                if img1 in visited:
                    continue
                group = []
                for img2, hash2 in hashes.items():
                    if img2 not in visited and hash1 - hash2 <= 2:
                        group.append(img2)
                        visited.add(img2)
                if len(group) > 1:
                    group_list.append(group)
                else:
                    st.session_state.unassigned_images.append(img1)
                progress_group.progress((idx + 1) / total_images, text=f"Grouping {idx + 1}/{total_images}")

            multi_image_groups = {i: g for i, g in enumerate(group_list)}
            st.session_state.ordered_groups = {**multi_image_groups}

            for group_id in multi_image_groups:
                st.session_state.uniware_sku_codes[group_id] = ""
                for img_path in multi_image_groups[group_id]:
                    st.session_state.image_level_sku_codes[img_path] = ""

            st.session_state.image_info = image_info
            st.session_state.hashes = hashes
            st.session_state.grouped = True

    st.subheader("Summary")
    st.write(f"Groups with >1 image: {len(st.session_state.ordered_groups)}")

    total_groups = list(st.session_state.ordered_groups.keys())

    if total_groups:
        current_group_num = total_groups[st.session_state.current_group_idx]
        imgs = st.session_state.ordered_groups[current_group_num]

        if len(imgs) == 0:
            st.session_state.current_group_idx = (st.session_state.current_group_idx + 1) % len(total_groups)
            st.rerun()

        st.markdown(f"### Group {current_group_num + 1} ({len(imgs)} images)")

        max_columns = 5
        cols = st.columns(min(len(imgs),max_columns))

        for idx, img_path in enumerate(imgs):
            with cols[idx % max_columns]:
                st.image(img_path, width=250)
                channel_name, channel_product_id, seller_sku, product_name, channel_code = st.session_state.image_info.get(img_path, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))
                st.caption(f"Channel Product Id: {channel_product_id}")
                st.caption(f"Seller SKU: {seller_sku}")
                st.caption(f"Product: {product_name}")
                st.caption(f"Channel: {channel_code}")
                
                st.text_input(f"Enter SKU for this image", key=f"img_sku_{img_path}",
                              value=st.session_state.image_level_sku_codes.get(img_path, ""))

                if st.button(f"Remove", key=f"remove_{img_path}"):
                    imgs.remove(img_path)
                    st.session_state.unassigned_images.append(img_path)

                    if len(imgs) == 1:
                        leftover_img = imgs.pop()
                        st.session_state.unassigned_images.append(leftover_img)
                        del st.session_state.ordered_groups[current_group_num]

                    st.rerun()

        st.text_input(f"Enter Uniware SKU Code for Group {current_group_num + 1}",
                      key=f"group_sku_{current_group_num}",
                      value=st.session_state.uniware_sku_codes.get(current_group_num, ""))

    if st.button("Confirm and Go to Next Group"):
        group_sku = st.session_state.get(f"group_sku_{current_group_num}", "")
        for img_path in imgs:
            img_level_sku = st.session_state.get(f"img_sku_{img_path}", "")
            if not img_level_sku:
                st.session_state.image_level_sku_codes[img_path] = group_sku
            else:
                if not group_sku:
                    st.session_state.image_level_sku_codes[img_path] = img_level_sku
                else:
                    st.session_state.image_level_sku_codes[img_path] = group_sku

        st.session_state.current_group_idx = (st.session_state.current_group_idx + 1) % len(total_groups)
        st.rerun()

    first_col, prev_col, skip_col, last_col = st.columns([1, 1, 1, 1])
    
    # Go to First Group button
    with first_col:
        if st.button("Go to First Group"):
            st.session_state.current_group_idx = 0
            st.rerun()
    
    with prev_col:
        if st.button("Previous Group"):
            st.session_state.current_group_idx = (st.session_state.current_group_idx - 1) % len(total_groups)
            st.rerun()

    with skip_col:
        if st.button("Skip to Next Group"):
            st.session_state.current_group_idx = (st.session_state.current_group_idx + 1) % len(total_groups)
            st.rerun()

    # Go to Last Group button
    with last_col:
        if st.button("Go to Last Group"):
            last_group_idx = len(total_groups) - 2  # Last group before the unassigned group
            st.session_state.current_group_idx = last_group_idx
            st.rerun()

    if st.button("Show/Hide Unassigned Group"):
        st.session_state.show_unassigned = not st.session_state.show_unassigned

    if st.session_state.show_unassigned:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Unassigned Group (-100)")
        
        # Limit columns to 5
        max_columns = 5
        cols = st.columns(min(len(st.session_state.unassigned_images), max_columns))

        for idx, img_path in enumerate(st.session_state.unassigned_images):
            with cols[idx % max_columns]:
                st.image(img_path, width=250)
                channel_name, channel_product_id, seller_sku, product_name, channel_code = st.session_state.image_info.get(img_path, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))
                st.caption(f"Channel Product Id: {channel_product_id} | Seller SKU: {seller_sku} | Product: {product_name} | Channel: {channel_code}")
                uniware_sku_code = st.text_input(f"Uniware SKU Code for {img_path}", key=f"uniware_sku_{img_path}",
                                             value=st.session_state.image_level_sku_codes.get(img_path, ""))

                if uniware_sku_code:
                    st.session_state.image_level_sku_codes[img_path] = uniware_sku_code
                    st.session_state.unassigned_images = [img for img in st.session_state.unassigned_images if img != img_path]
                    st.session_state.unassigned_images.append(img_path)
        st.write("The Uniware SKU Codes will be added to the export data as soon as you type and press enter.")
                
    # Add a button to refresh and generate the export data
    if st.button("Refresh Export Data"):
        st.session_state.export_data = []
        for group_id, img_list in st.session_state.ordered_groups.items():
            for img in img_list:
                channel_name, channel_product_id, seller_sku, product_name, channel_code = st.session_state.image_info.get(img, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))
                image_sku = st.session_state.image_level_sku_codes.get(img, "")
                if not image_sku:
                    image_sku = st.session_state.uniware_sku_codes.get(group_id, "")
                st.session_state.export_data.append({
                    "Channel Name": channel_name,
                    "Channel Product Id": channel_product_id,
                    "Seller SKU Code": seller_sku,
                    "Uniware SKU Code": image_sku,
                    "Product name": product_name
                })

        for img in st.session_state.unassigned_images:
            channel_name, channel_product_id, seller_sku, product_name, channel_code = st.session_state.image_info.get(img, ("Unknown", "Unknown", "Unknown", "Unknown", "Unknown"))
            st.session_state.export_data.append({
                "Channel Name": channel_name,
                "Channel Product Id": channel_product_id,
                "Seller SKU Code": seller_sku,
                "Uniware SKU Code": "",
                "Product name": product_name
            })

        result_df = pd.DataFrame(st.session_state.export_data)
        # Convert the DataFrame to a byte object in Excel format
        to_excel = io.BytesIO()
        with pd.ExcelWriter(to_excel, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name="Grouped Images")
        to_excel.seek(0)  # Seek to the beginning of the byte stream

        # Display the dataframe and provide a download button
        st.dataframe(result_df)
        st.download_button(
            "Download Final Excel",
            to_excel,
            file_name="final_groups.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
