import numpy as np
import cv2
import os
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_sift_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=5,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )

    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


class VocabularyTree:
    def __init__(self, branch_factor=8, depth=4):
        self.branch_factor = branch_factor
        self.depth = depth
        self.nodes = {}
        self.leaf_inverted_files = defaultdict(list)
        self.image_weights = defaultdict(dict)

    def build_tree(self, descriptors, n_training_cycles=20):
        print("Building vocabulary tree...")
        current_descriptors = descriptors

        for level in range(self.depth):
            print(f"Processing level {level + 1}/{self.depth}")
            if len(current_descriptors) < self.branch_factor:
                break

            kmeans = KMeans(
                n_clusters=min(self.branch_factor, len(current_descriptors)),
                n_init=n_training_cycles,
                max_iter=300
            )
            labels = kmeans.fit_predict(current_descriptors)
            self.nodes[level] = kmeans

            if level < self.depth - 1:
                new_descriptors = []
                for i in range(self.branch_factor):
                    cluster_descriptors = current_descriptors[labels == i]
                    if len(cluster_descriptors) > self.branch_factor:
                        new_descriptors.extend(cluster_descriptors)
                current_descriptors = np.array(new_descriptors) if new_descriptors else np.array([])

    def get_path(self, descriptor):
        path = []
        current_descriptor = descriptor.reshape(1, -1)

        for level in range(self.depth):
            if level not in self.nodes:
                break
            kmeans = self.nodes[level]
            cluster_id = kmeans.predict(current_descriptor)[0]
            path.append(cluster_id)

        return tuple(path)

    def add_image(self, image_id, descriptors):
        if descriptors is None or len(descriptors) == 0:
            return

        paths = [self.get_path(desc) for desc in descriptors]

        path_counts = defaultdict(int)
        for path in paths:
            path_counts[path] += 1

        total_descriptors = len(paths)
        normalized_counts = {path: count / total_descriptors
                             for path, count in path_counts.items()}

        self.image_weights[image_id] = normalized_counts

        for path in normalized_counts:
            self.leaf_inverted_files[path].append(image_id)

    def compute_similarity(self, query_weights, db_weights):
        similarity = 0
        all_paths = set(query_weights.keys()) | set(db_weights.keys())

        for path in all_paths:
            query_weight = query_weights.get(path, 0)
            db_weight = db_weights.get(path, 0)
            similarity += min(query_weight, db_weight)

        return similarity

    def query_image(self, descriptors, top_k=5):

        if descriptors is None or len(descriptors) == 0:
            return []

        paths = [self.get_path(desc) for desc in descriptors]
        query_counts = defaultdict(int)
        for path in paths:
            query_counts[path] += 1

        total_descriptors = len(paths)
        query_weights = {path: count / total_descriptors
                         for path, count in query_counts.items()}

        scores = []
        for image_id in self.image_weights:
            similarity = self.compute_similarity(query_weights, self.image_weights[image_id])
            scores.append((image_id, similarity))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


def display_results(query_path, results, save_path=None):

    plt.figure(figsize=(15, 10))

    query_img = cv2.imread(query_path)
    if query_img is None:
        print(f"Error: Could not read query image: {query_path}")
        return
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 3, 1)
    plt.imshow(query_img)
    plt.title('Query Image', fontsize=12)
    plt.axis('off')

    for idx, (path, score) in enumerate(results[:5]):

        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not read result image: {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 3, idx + 2)
        plt.imshow(img)
        plt.title(f'Match {idx + 1}\nScore: {score:.3f}', fontsize=10)
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def query_and_visualize(vtree, query_image_path, save_path=None):
    was_training = query_image_path in vtree.training_paths
    print(f"\nQuery image was{' ' if was_training else ' not '}used in training")

    descriptors = extract_sift_features(query_image_path)
    if descriptors is not None:
        results = vtree.query_image(descriptors, top_k=5)
        if results:

            query_in_results = any(query_image_path == path for path, _ in results)
            print(f"Query image was{' ' if query_in_results else ' not '}found in results")

            display_results(query_image_path, results, save_path)
            return results
    return None


def build_image_retrieval_system(image_folder, sample_size=2000):

    vtree = VocabularyTree(branch_factor=10, depth=6)

    image_paths = list(Path(image_folder).glob('*.jpg'))
    print(f"Found {len(image_paths)} images in dataset")

    training_paths = np.random.choice(image_paths, min(sample_size, len(image_paths)), replace=False)
    vtree.training_paths = set(str(path) for path in training_paths)  # Store for reference

    print("Extracting features for training...")
    training_descriptors = []

    for path in tqdm(training_paths):
        descriptors = extract_sift_features(str(path))
        if descriptors is not None:
            if len(descriptors) > 100:
                indices = np.random.choice(len(descriptors), 100, replace=False)
                descriptors = descriptors[indices]
            training_descriptors.extend(descriptors)

    if len(training_descriptors) == 0:
        raise Exception("No features could be extracted from the training images")

    training_descriptors = np.array(training_descriptors)
    print(f"Extracted {len(training_descriptors)} training descriptors")

    vtree.build_tree(training_descriptors)

    print("Adding all images to database...")
    for path in tqdm(image_paths):
        descriptors = extract_sift_features(str(path))
        vtree.add_image(str(path), descriptors)

    return vtree


def process_batch(paths, vtree):
    results = []
    for path in paths:
        descriptors = extract_sift_features(str(path))
        if descriptors is not None:
            vtree.add_image(str(path), descriptors)
    return results


def crop_image(image_path, crop_ratio=0.8):
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)
    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2

    cropped = image[start_h:start_h + new_h, start_w:start_w + new_w]

    return cropped


def extract_sift_features_from_array(image):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=5,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )

    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def calculate_recalls(results, query_path, k_values=[1, 5, 10]):
    query_base = os.path.basename(query_path).split('_')[0]
    total_relevant = sum(1 for path, _ in vtree.image_weights.items()
                         if os.path.basename(path).split('_')[0] == query_base)

    recalls = {}
    for k in k_values:
        top_k_results = results[:k]
        relevant_retrieved = sum(1 for path, _ in top_k_results
                                 if os.path.basename(path).split('_')[0] == query_base)
        recalls[k] = relevant_retrieved / total_relevant if total_relevant > 0 else 0

    return recalls


def evaluate_system(vtree, image_folder, num_queries=100, crop_ratio=0.8):
    image_paths = list(Path(image_folder).glob('*.jpg'))
    query_paths = np.random.choice(image_paths, num_queries, replace=False)
    all_recalls = defaultdict(list)

    print("Processing queries...")
    for query_path in tqdm(query_paths):
        cropped_img = crop_image(str(query_path), crop_ratio)
        if cropped_img is None:
            continue
        descriptors = extract_sift_features_from_array(cropped_img)
        if descriptors is None:
            continue
        results = vtree.query_image(descriptors, top_k=10)
        recalls = calculate_recalls(results, str(query_path))

        for k, value in recalls.items():
            all_recalls[k].append(value)

    mean_recalls = {k: np.mean(values) for k, values in all_recalls.items()}

    return mean_recalls


def query_with_crop(vtree, query_path, crop_ratio=0.8, save_path=None):
    cropped_img = crop_image(query_path, crop_ratio)
    if cropped_img is None:
        print("Error: Could not crop image")
        return None

    descriptors = extract_sift_features_from_array(cropped_img)
    if descriptors is None:
        print("Error: Could not extract features")
        return None

    results = vtree.query_image(descriptors, top_k=5)

    if results:
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        plt.imshow(cropped_img_rgb)
        plt.title('Cropped Query', fontsize=12)
        plt.axis('off')

        for idx, (path, score) in enumerate(results):
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.subplot(2, 3, idx + 2)
            plt.imshow(img)
            plt.title(f'Match {idx + 1}\nScore: {score:.3f}', fontsize=10)
            plt.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    return results

if __name__ == "__main__":
    image_folder = "flickr8k"
    query_path = "flickr8k/667626_18933d713e.jpg"

    if not os.path.exists('vocabulary_tree_full.pkl'):
        print("Building new vocabulary tree for full dataset...")
        vtree = build_image_retrieval_system(image_folder)
        print("Saving vocabulary tree...")
        with open('vocabulary_tree_full.pkl', 'wb') as f:
            pickle.dump(vtree, f)
    else:
        print("Loading existing vocabulary tree...")
        with open('vocabulary_tree_full.pkl', 'rb') as f:
            vtree = pickle.load(f)

    # results = query_and_visualize(vtree, query_path, 'retrieval_results_full.png')
    #
    # if results:
    #     print("\nTop 5 matching images:")
    #     for path, score in results:
    #         print(f"Path: {path}, Score: {score:.4f}")
    print("\nTesting single query with cropping...")
    results = query_with_crop(vtree, query_path, crop_ratio=0.75,
                              save_path='retrieval_results_cropped.png')

    if results:
        print("\nTop 5 matching images:")
        for path, score in results:
            print(f"Path: {path}, Score: {score:.4f}")

    print("\nEvaluating system performance...")
    mean_recalls = evaluate_system(vtree, image_folder, num_queries=500, crop_ratio=0.85)

    print("\nMean Recall Values:")
    for k, value in sorted(mean_recalls.items()):
        print(f"Recall@{k}: {value:.4f}")

