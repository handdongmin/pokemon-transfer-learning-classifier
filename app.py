from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HERO_IMAGE = Path("assets/pokemon_hero.png")

MODEL_OPTIONS = {
    "ResNet18 - 전체 파인튜닝": "models/resnet18_finetune.pth",
    "EfficientNet-B0 - 전체 파인튜닝": "models/efficientnet_finetune.pth",
    "ResNet18 - 특징 추출": "models/resnet18_frozen.pth",
    "MobileNetV2 - 특징 추출": "models/mobilenet_frozen.pth",
}

MODEL_DESCRIPTIONS = {
    "ResNet18 - 전체 파인튜닝": "ImageNet 사전학습 가중치를 시작점으로 모든 layer를 학습한 최고 정확도 모델입니다. 데모 테스트에 가장 추천합니다.",
    "EfficientNet-B0 - 전체 파인튜닝": "효율적인 backbone 전체를 파인튜닝한 높은 recall 모델입니다.",
    "ResNet18 - 특징 추출": "ImageNet 사전학습 가중치를 사용하고 backbone은 고정한 실험입니다. 성능 비교용 baseline입니다.",
    "MobileNetV2 - 특징 추출": "가벼운 backbone을 고정하고 classifier head만 학습한 빠른 실험입니다. 성능 비교용 baseline입니다.",
}


st.set_page_config(page_title="포켓몬 분류기", page_icon="P", layout="wide")


def inject_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7fbff 0%, #fff8e8 48%, #ffffff 100%);
        }
        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 3rem;
            max-width: 1240px;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h2, h3, p, div, label, span {
            word-break: keep-all;
            overflow-wrap: break-word;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e8edf5;
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
            box-shadow: 0 8px 24px rgba(30, 45, 80, 0.06);
        }
        [data-testid="stMetricLabel"] {
            white-space: nowrap;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.85rem;
        }
        .hero-title {
            font-size: 1.95rem;
            font-weight: 800;
            line-height: 1.55;
            color: #172033;
            margin: 0 0 0.45rem;
            padding-top: 0.35rem;
            padding-bottom: 0.05rem;
            overflow: visible;
        }
        .hero-subtitle {
            color: #4d5b73;
            font-size: 0.98rem;
            line-height: 1.75;
            margin-bottom: 1.1rem;
        }
        .hero-image img {
            border-radius: 8px;
        }
        .info-box {
            background: #ffffff;
            border: 1px solid #e8edf5;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(30, 45, 80, 0.06);
        }
        .prediction-card {
            background: #ffffff;
            border: 1px solid #e8edf5;
            border-radius: 8px;
            padding: 0.8rem 0.95rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 6px 18px rgba(30, 45, 80, 0.05);
        }
        .prediction-rank {
            color: #d94841;
            font-weight: 800;
            font-size: 0.9rem;
        }
        .prediction-name {
            color: #172033;
            font-weight: 750;
            font-size: 1.08rem;
            margin-top: 0.15rem;
        }
        .prediction-score {
            color: #526174;
            font-size: 0.92rem;
            margin-top: 0.1rem;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .hero-title {
                font-size: 1.7rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_model(exp_name, num_classes):
    if exp_name in ["resnet18_frozen", "resnet18_finetune"]:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif exp_name == "mobilenet_frozen":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif exp_name == "efficientnet_finetune":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown experiment name: {exp_name}")

    return model


@st.cache_resource
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    exp_name = checkpoint["exp_name"]

    model = build_model(exp_name, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def show_prediction(rank, class_name, confidence):
    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-rank">TOP {rank}</div>
            <div class="prediction-name">{class_name}</div>
            <div class="prediction-score">확신도 {confidence * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence)


inject_style()

left, right = st.columns([1.1, 1], gap="medium")

with left:
    st.markdown('<div class="hero-title">포켓몬 이미지 분류 데모</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-subtitle">
        학습된 transfer learning 모델을 선택하고 포켓몬 이미지를 업로드하면,
        모델이 예측한 이름과 Top-5 확신도를 보여줍니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("분류 클래스", "150")
    metric_cols[1].metric("실험 설정", "4개")
    metric_cols[2].metric("실행 장치", DEVICE.upper())

with right:
    if HERO_IMAGE.exists():
        st.markdown('<div class="hero-image">', unsafe_allow_html=True)
        st.image(str(HERO_IMAGE), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("대표 이미지 파일을 찾지 못했습니다. 예측 기능은 그대로 사용할 수 있습니다.")

st.divider()

control_col, result_col = st.columns([1, 1.15], gap="medium")

with control_col:
    st.subheader("테스트 설정")
    selected_model = st.selectbox("사용할 모델", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[selected_model]

    st.markdown(
        f"""
        <div class="info-box">
        <b>모델 설명</b><br>
        {MODEL_DESCRIPTIONS[selected_model]}<br><br>
        <b>체크포인트</b><br>
        {model_path}
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "포켓몬 이미지 업로드",
        type=["jpg", "jpeg", "png", "webp"],
        help="JPG, PNG, WEBP 이미지를 사용할 수 있습니다.",
    )

with result_col:
    st.subheader("예측 결과")

    if not Path(model_path).exists():
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        st.stop()

    if uploaded_file is None:
        st.info("왼쪽에서 이미지를 업로드하면 예측 결과가 여기에 표시됩니다.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        preview_col, pred_col = st.columns([0.85, 1], gap="medium")

        with preview_col:
            st.image(image, caption="업로드한 이미지", use_container_width=True)

        with pred_col:
            with st.spinner("모델이 이미지를 분석하는 중입니다..."):
                model, class_names = load_model(model_path)
                x = preprocess(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(x)
                    probabilities = torch.softmax(output, dim=1)
                    top_probs, top_idxs = torch.topk(probabilities, 5)

            for rank, (probability, class_idx) in enumerate(zip(top_probs[0], top_idxs[0]), start=1):
                show_prediction(rank, class_names[class_idx.item()], probability.item())
