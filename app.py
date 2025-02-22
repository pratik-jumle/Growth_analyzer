import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define PlantAnalyzer class
class PlantAnalyzer:
    def __init__(self, week1_image, week2_image):
        """Initialize with image arrays instead of file paths."""
        self.week1_image = week1_image
        self.week2_image = week2_image
        if self.week1_image is None or self.week2_image is None:
            raise ValueError("One or both images failed to load.")

    def segment_plant(self, image):
        """Segment the plant from the background using color thresholding."""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        segmented = cv2.bitwise_and(image, image, mask=mask)
        return segmented, mask

    def measure_height(self, mask):
        """Measure plant height in pixels."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            _, y, _, h = cv2.boundingRect(largest_contour)
            return h
        return 0

    def estimate_leaf_count(self, mask):
        """Estimate leaf count with contour filtering."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 30
        max_area = 5000
        leaves = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    leaves.append(c)
        return len(leaves)

    def measure_leaf_quality(self, image, mask):
        """Measure leaf quality based on greenness."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        plant_pixels = hsv[mask > 0]
        if plant_pixels.size > 0:
            avg_hue = np.mean(plant_pixels[:, 0])
            avg_saturation = np.mean(plant_pixels[:, 1])
            quality_score = avg_saturation * (1 - abs(avg_hue - 60) / 60)
            return quality_score
        return 0

    def analyze_growth(self):
        """Analyze growth between Week 1 and Week 2."""
        week1_segmented, week1_mask = self.segment_plant(self.week1_image)
        week2_segmented, week2_mask = self.segment_plant(self.week2_image)

        height1 = self.measure_height(week1_mask)
        height2 = self.measure_height(week2_mask)
        leaf_count1 = self.estimate_leaf_count(week1_mask)
        leaf_count2 = self.estimate_leaf_count(week2_mask)
        quality1 = self.measure_leaf_quality(self.week1_image, week1_mask)
        quality2 = self.measure_leaf_quality(self.week2_image, week2_mask)

        height_growth = (height2 - height1) / height1 * 100 if height1 > 0 else (0 if height2 == 0 else 100)
        leaf_count_growth = (leaf_count2 - leaf_count1) / leaf_count1 * 100 if leaf_count1 > 0 else (0 if leaf_count2 == 0 else 100)
        quality_growth = (quality2 - quality1) / quality1 * 100 if quality1 > 0 else (0 if quality2 == 0 else 100)

        overall_growth = (height_growth * 0.4 + leaf_count_growth * 0.3 + quality_growth * 0.3)
        status = "Significant Growth" if overall_growth > 15 else "Not Significant Growth"

        return {
            'height_week1': height1, 'height_week2': height2, 'height_growth_percentage': height_growth,
            'leaf_count_week1': leaf_count1, 'leaf_count_week2': leaf_count2, 'leaf_count_growth_percentage': leaf_count_growth,
            'leaf_quality_week1': quality1, 'leaf_quality_week2': quality2, 'leaf_quality_growth_percentage': quality_growth,
            'overall_growth_percentage': overall_growth, 'growth_status': status
        }

# Visualization functions
def plot_comparison_bar(growth_result):
    """Plot Week 1 vs Week 2 metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Height (pixels)', 'Leaf Count', 'Leaf Quality']
    week1_values = [growth_result['height_week1'], growth_result['leaf_count_week1'], growth_result['leaf_quality_week1']]
    week2_values = [growth_result['height_week2'], growth_result['leaf_count_week2'], growth_result['leaf_quality_week2']]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, week1_values, width, label='Week 1', color='lightblue')
    ax.bar(x + width/2, week2_values, width, label='Week 2', color='lightgreen')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Plant Metrics: Week 1 vs Week 2')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_growth_percentage_bar(growth_result):
    """Plot growth percentages with threshold."""
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics = ['Height Growth', 'Leaf Count Growth', 'Leaf Quality Growth', 'Overall Growth']
    percentages = [
        growth_result['height_growth_percentage'],
        growth_result['leaf_count_growth_percentage'],
        growth_result['leaf_quality_growth_percentage'],
        growth_result['overall_growth_percentage']
    ]
    colors = ['skyblue' if p <= 15 else 'limegreen' for p in percentages]
    bars = ax.bar(metrics, percentages, color=colors)
    ax.axhline(y=15, color='red', linestyle='--', label='15% Threshold')
    ax.set_xlabel('Growth Metrics')
    ax.set_ylabel('Growth Percentage (%)')
    ax.set_title(f'Growth Percentages from Week 1 to Week 2\nStatus: {growth_result["growth_status"]}')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
    if growth_result['overall_growth_percentage'] < 15:
        ax.text(0.5, 0.95, '*** DIAGNOSE DISEASE FOR YOUR PLANT ***',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=20, fontweight='bold', color='red', bbox=dict(facecolor='yellow', alpha=0.5))
    plt.tight_layout()
    return fig

def plot_growth_trend(growth_result):
    """Plot growth trends over weeks."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Height', 'Leaf Count', 'Leaf Quality']
    week1_values = [growth_result['height_week1'], growth_result['leaf_count_week1'], growth_result['leaf_quality_week1']]
    week2_values = [growth_result['height_week2'], growth_result['leaf_count_week2'], growth_result['leaf_quality_week2']]
    ax.plot(['Week 1', 'Week 2'], [week1_values[0], week2_values[0]], marker='o', label='Height (pixels)', color='blue')
    ax.plot(['Week 1', 'Week 2'], [week1_values[1], week2_values[1]], marker='o', label='Leaf Count', color='green')
    ax.plot(['Week 1', 'Week 2'], [week1_values[2], week2_values[2]], marker='o', label='Leaf Quality', color='red')
    ax.set_xlabel('Week')
    ax.set_ylabel('Values')
    ax.set_title('Growth Trend from Week 1 to Week 2')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def show_images(analyzer):
    """Display original images and masks."""
    _, week1_mask = analyzer.segment_plant(analyzer.week1_image)
    _, week2_mask = analyzer.segment_plant(analyzer.week2_image)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    axes[0, 0].set_title("Week 1 Image")
    axes[0, 0].imshow(cv2.cvtColor(analyzer.week1_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].axis('off')
    axes[0, 1].set_title("Week 1 Mask")
    axes[0, 1].imshow(week1_mask, cmap='gray')
    axes[0, 1].axis('off')
    axes[1, 0].set_title("Week 2 Image")
    axes[1, 0].imshow(cv2.cvtColor(analyzer.week2_image, cv2.COLOR_BGR2RGB))
    axes[1, 0].axis('off')
    axes[1, 1].set_title("Week 2 Mask")
    axes[1, 1].imshow(week2_mask, cmap='gray')
    axes[1, 1].axis('off')
    plt.tight_layout()
    return fig

# Streamlit app
st.title("Plant Growth Analyzer")
st.write("Upload images for Week 1 and Week 2 to analyze plant growth.")

uploaded_week1 = st.file_uploader("Upload Week 1 Image", type=["jpg", "jpeg", "png"])
uploaded_week2 = st.file_uploader("Upload Week 2 Image", type=["jpg", "jpeg", "png"])

if uploaded_week1 and uploaded_week2:
    if st.button("Analyze"):
        try:
            # Read uploaded images
            week1_image = cv2.imdecode(np.frombuffer(uploaded_week1.read(), np.uint8), cv2.IMREAD_COLOR)
            week2_image = cv2.imdecode(np.frombuffer(uploaded_week2.read(), np.uint8), cv2.IMREAD_COLOR)

            if week1_image is None or week2_image is None:
                st.error("Failed to load one or both images. Please check the files.")
            else:
                # Analyze growth
                with st.spinner("Analyzing images..."):
                    analyzer = PlantAnalyzer(week1_image, week2_image)
                    growth_result = analyzer.analyze_growth()

                # Improved Results Display (Column-Based Layout)
                st.write("### Plant Growth Summary")
                st.markdown(f"**Overall Growth**: {growth_result['overall_growth_percentage']:.1f}%", unsafe_allow_html=True)
                status_color = "green" if growth_result['overall_growth_percentage'] > 15 else "orange"
                st.markdown(f"<h4 style='color:{status_color};'>Status: {growth_result['growth_status']}</h4>", unsafe_allow_html=True)
                if growth_result['overall_growth_percentage'] < 15:
                    st.warning("Growth is below 15%. Your Plant Need Disease Check-up")

                with st.expander("View Detailed Metrics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Height**")
                        st.write(f"Week 1: {growth_result['height_week1']} pixels")
                        st.write(f"Week 2: {growth_result['height_week2']} pixels")
                        st.write(f"Growth: {growth_result['height_growth_percentage']:.1f}%")
                    with col2:
                        st.write("**Leaf Count**")
                        st.write(f"Week 1: {growth_result['leaf_count_week1']}")
                        st.write(f"Week 2: {growth_result['leaf_count_week2']}")
                        st.write(f"Growth: {growth_result['leaf_count_growth_percentage']:.1f}%")
                    with col3:
                        st.write("**Leaf Quality**")
                        st.write(f"Week 1: {growth_result['leaf_quality_week1']:.1f}")
                        st.write(f"Week 2: {growth_result['leaf_quality_week2']:.1f}")
                        st.write(f"Growth: {growth_result['leaf_quality_growth_percentage']:.1f}%")
                st.caption("Note: Leaf Quality is a score based on greenness.")

                # Display visualizations
                st.write("### Visualizations")
                st.write("**Comparison of Plant Metrics**")
                fig = plot_comparison_bar(growth_result)
                st.pyplot(fig)
                plt.close(fig)

                st.write("**Growth Percentages**")
                fig = plot_growth_percentage_bar(growth_result)
                st.pyplot(fig)
                plt.close(fig)

                st.write("**Growth Trend**")
                fig = plot_growth_trend(growth_result)
                st.pyplot(fig)
                plt.close(fig)

                st.write("**Images and Segmentation Masks**")
                fig = show_images(analyzer)
                st.pyplot(fig)
                plt.close(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")