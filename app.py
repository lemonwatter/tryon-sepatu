import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
# Import SEMUA lapisan yang digunakan di Colab agar netG bisa direkonstruksi
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Activation, BatchNormalization, Dropout, Concatenate, Conv2DTranspose
import numpy as np
from PIL import Image
import os
import glob
import gdown # Library untuk Google Drive

# ==============================================================================
# KONFIGURASI DAN UTILITAS
# ==============================================================================
IMG_SIZE = 256
MODEL_G_PATH = 'models/pix2pix_tryon_G.h5'
GDRIVE_G_ID = '1UR586B33eOGJgKWCcrqm80KGYefvxK1c' # ID Model Generator

# --- 1. Re-Definisi Arsitektur Model (Wajib Sama dengan Training) ---

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(BatchNormalization())
    result.add(LeakyReLU(0.2))
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    result.add(BatchNormalization())
    if apply_dropout:
        result.add(Dropout(0.5))
    result.add(Activation('relu'))
    return result

# Menggunakan st.cache_resource untuk menampung arsitektur
@st.cache_resource
def GeneratorUNet(input_shape=(IMG_SIZE, IMG_SIZE, 7), output_channels=3):
    inputs = Input(shape=input_shape)
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), downsample(128, 4), downsample(256, 4),
        downsample(512, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4),
        downsample(512, 4, apply_batchnorm=False),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True), upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True), upsample(512, 4), upsample(256, 4),
        upsample(128, 4), upsample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                           kernel_initializer=initializer, activation='tanh')
    x = inputs; skips = []
    for down in down_stack:
        x = down(x); skips.append(x)
    x = up_stack[0](skips[-1])
    for up, skip in zip(up_stack[1:], reversed(skips[:-1])):
        x = Concatenate()([x, skip]); x = up(x)
    x = Concatenate()([x, skips[0]])
    x = last(x)
    return Model(inputs=inputs, outputs=x, name='Generator')

# --- 2. Fungsi Download Model (Model Besar) ---

@st.cache_resource
def download_model_from_gdrive(file_id, output_path):
    # Buat folder models/ jika belum ada
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Cek apakah model sudah ada (untuk mencegah download berulang setelah deploy)
    if os.path.exists(output_path):
        st.info(f"✅ Model Generator sudah ada. Melewati unduhan.")
        return True

    st.warning("⏳ Mengunduh model Generator besar dari Google Drive (~200MB+). Proses ini bisa memakan waktu beberapa menit saat deployment pertama.")
    
    try:
        # Unduh file berdasarkan ID
        gdown.download(id=file_id, output=output_path, quiet=False, fuzzy=True)
        
        if os.path.exists(output_path):
             st.success("🎉 Model Generator berhasil diunduh!")
             return True
        else:
             st.error("❌ Gagal mengunduh model. Cek ID Drive dan izin akses publik.")
             return False
             
    except Exception as e:
        st.error(f"❌ Error saat mengunduh model: {e}")
        st.stop()

# --- 3. Fungsi Load Model ---

@st.cache_resource
def load_generator_model():
    # Panggil fungsi download
    if not download_model_from_gdrive(GDRIVE_G_ID, MODEL_G_PATH):
        st.error("Aplikasi dihentikan karena gagal mengunduh model.")
        st.stop()

    # Load bobot
    netG = GeneratorUNet()
    try:
        netG.load_weights(MODEL_G_PATH)
        return netG
    except Exception as e:
        st.error(f"❌ Gagal memuat bobot model: {e}")
        st.stop()

# --- 4. Fungsi Preprocessing & Utility ---

def load_and_preprocess(img_data, is_mask=False):
    # img_data bisa berupa path atau file yang di-upload
    if isinstance(img_data, str):
        img = Image.open(img_data)
    else: # File Upload Streamlit
        img = Image.open(img_data)

    if is_mask:
        img = img.convert('L').resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img)[..., np.newaxis].astype(np.float32)
    else:
        img = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img).astype(np.float32)

    return (img / 127.5) - 1.0 # Normalisasi ke [-1, 1]

def get_assets(folder):
    return sorted(glob.glob(os.path.join('assets', folder, '*')))


# ==============================================================================
# HALAMAN UTAMA STREAMLIT (UI)
# ==============================================================================

st.set_page_config(layout="wide", page_title="Virtual Try-On Sepatu GAN")

st.title("👟 Virtual Try-On Sepatu: Pix2Pix GAN")
st.markdown("Pilih sepatu, masukkan citra kaki, dan klik **Terapkan Virtual Try-On**.")

# Memuat Model (Memicu download jika belum ada)
netG = load_generator_model()
st.sidebar.success("✅ Model Generator Siap Digunakan.")

# Persiapan Aset
shoe_assets = get_assets('shoes')
feet_assets = get_assets('feet')
if not shoe_assets or not feet_assets:
    st.error("❌ Pastikan folder assets/shoes/ dan assets/feet/ terisi gambar.")
    st.stop()

# --------------------------------------------------
# PILIHAN ASET & INPUT KAKI
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.header("1. Pilih Sepatu (IC)")
    shoe_options = [os.path.basename(p) for p in shoe_assets]
    selected_shoe_name = st.selectbox("Pilih Sepatu:", shoe_options, index=0)
    selected_shoe_path = os.path.join('assets', 'shoes', selected_shoe_name)
    st.image(selected_shoe_path, caption=selected_shoe_name, width=256)

with col2:
    st.header("2. Input Kaki (IA)")
    input_method = st.radio("Pilih Metode Input Kaki:", ("Pilih dari Galeri", "Unggah Citra Baru"))

    input_feet_data = None # Bisa berupa path atau file object

    if input_method == "Pilih dari Galeri":
        feet_options = [os.path.basename(p) for p in feet_assets]
        selected_feet_name = st.selectbox("Pilih Bentuk Kaki:", feet_options, index=0)
        input_feet_data = os.path.join('assets', 'feet', selected_feet_name)
        st.image(input_feet_data, caption=selected_feet_name, width=256)
    else:
        uploaded_file = st.file_uploader("Unggah Citra Kaki (JPG, PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            input_feet_data = uploaded_file
            st.image(uploaded_file, caption="Citra Kaki yang Diunggah", width=256)

# --------------------------------------------------
# TOMBOL TRY-ON & INFERENCE
# --------------------------------------------------
st.markdown("---")
if st.button("✨ Terapkan Virtual Try-On"):
    if input_feet_data is not None:
        with st.spinner("Sedang memproses dan menghasilkan citra virtual try-on..."):
            try:
                # Muat dan Preprocessing Sepatu (IC)
                ic_img_np = load_and_preprocess(selected_shoe_path, is_mask=False) # (H, W, 3)

                # Muat dan Preprocessing Kaki (IA)
                ia_img_np = load_and_preprocess(input_feet_data, is_mask=False) # (H, W, 3)

                # SIMULASI MASKER (IM) - Model Anda butuh 7 Channel: [IA (3), IC (3), IM (1)]
                # Kita gunakan masker nol karena tidak ada model segmentasi untuk sepatu lama.
                im_img_np = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
                im_img_np = (im_img_np / 127.5) - 1.0 # Normalisasi ke [-1, 1]

                # Gabungkan Input ke (256, 256, 7)
                input_tensor_7ch = np.concatenate([ia_img_np, ic_img_np, im_img_np], axis=-1)

                # Tambahkan dimensi Batch
                input_tensor_4d = np.expand_dims(input_tensor_7ch, axis=0) # (1, H, W, 7)

                # INFERENCE MODEL
                fake_image_tf = netG(input_tensor_4d, training=False)
                
                # Konversi hasil ke [0, 1] untuk ditampilkan
                fake_image_np = (fake_image_tf.numpy()[0] * 0.5) + 0.5
                fake_image_display = np.clip(fake_image_np, 0, 1)

                # Tampilkan Hasil
                st.markdown("### Hasil Virtual Try-On")
                st.image(fake_image_display, caption="Hasil Generasi Sepatu Baru", use_column_width=True)

            except Exception as e:
                st.error(f"Terjadi Kesalahan saat Inference: {e}")
                st.info("Pastikan citra input memiliki resolusi yang wajar dan model berhasil dimuat.")
    else:
        st.warning("Mohon pilih atau unggah citra kaki terlebih dahulu.")
# --------------------------------------------------