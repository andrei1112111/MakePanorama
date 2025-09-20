import os
import cv2
import numpy as np
import argparse
from pathlib import Path


class PanoramaGenerator():
    def load_images(self, folder_path: str, image_format: str):
        if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
            return TypeError("Директория с изображениями не найдена")
        
        folder = Path(folder_path)
        n_imgs = len(list(folder.iterdir()))  # количество файлов в папке

        # Загружаем список изображений
        images = [cv2.imread(f"{folder_path}/{i}.{image_format}") for i in range(1, n_imgs)]  # img1.jpg, ...

        # Проверка загрузки
        images = [img for img in images if img is not None]
        if not images:
            print(images)
            raise ValueError("Не удалось загрузить изображения!")

        return images


    def make_panorama_SIFT(self, images: list):
        """
        Сшивает список изображений в одну панораму с помощью OpenCV SIFT.
        :param images: список изображений (numpy массивы)
        :return: панорама (numpy массив) или None, если ошибка
        """

        def stitch_pair(img1, img2):
            """
            Сшивает два изображения в одну панораму с помощью SIFT + FLANN.
            """
            # === SIFT === извелаем ключевые точки
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # === FLANN matcher === сравниваем ключевые точки и ищем пары похожих
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)

            # === правило Лоу === оставляем только пары где точки достаточно похожи
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 4:
                raise Exception("Недостаточно совпадений для сшивания")

            # === Гомография === поиск говографии для правильного составления панорамы
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # === Сшивание === сшиваем картинки
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            pts1_transformed = cv2.perspectiveTransform(pts1, H)

            pts_all = np.concatenate((pts1_transformed, np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)), axis=0)
            [xmin, ymin] = np.int32(pts_all.min(axis=0).ravel())
            [xmax, ymax] = np.int32(pts_all.max(axis=0).ravel())

            translation = [-xmin, -ymin]
            T = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

            result = cv2.warpPerspective(img1, T @ H, (xmax - xmin, ymax - ymin))
            result[translation[1]:h2 + translation[1], translation[0]:w2 + translation[0]] = img2

            return result

        panorama = images[0]
        for i in range(1, len(images)):
            print(f"\rСшивание {i}-го изображения...")
            panorama = stitch_pair(panorama, images[i])
            if panorama is None:
                break

        return panorama


    def make_panorama_Stitcher(self, images: list):
        """
        Сшивает список изображений в одну панораму с помощью OpenCV Stitcher.
        :param images: список изображений (numpy массивы)
        :return: панорама (numpy массив) или None, если ошибка
        """

        # Проверим входные данные
        images = [img for img in images if img is not None]
        if len(images) < 2:
            raise Exception("Нужно минимум 2 изображения для панорамы.")
            return None

        # Создаём stitcher
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

        # Пытаемся сшить
        status, pano = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            return pano
        elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            raise Exception("Недостаточно изображений для панорамы")
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            raise Exception("Не удалось найти гомографию")
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            raise Exception("Ошибка подгонки параметров камеры")
        else:
            raise Exception(f"Неизвестная ошибка (код {status})")


def parse_args():
    """
    парсер аргументов командной строки
    """
    parser = argparse.ArgumentParser(description="генерация панорамы через ключевые точки SIFT")

    parser.add_argument(
        "--folder", "-f", type=str, required=True,
        help="папка с картинками понорамы содержащая [1.jpg ... n.jpg]"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=False, default="panorama.jpg",
        help="Имя файла для готовой панорамы. panorama.jpg по умолчанию"
    )
    parser.add_argument(
        "--mode", "-m", type=str, required=False, default="Stitcher",
        help="Алгоритм создания панорамы на выбор SIFT / Stitcher(по умолчанию)"
    )
    parser.add_argument(
        "--image_format", "-i", type=str, required=False, default="jpg",
        help="Формат изображений. jpg по умолчанию"
    )

    args = parser.parse_args()

    if args.mode not in ["SIFT", "Stitcher"]:
        raise TypeError("Заданный алгоритм не поддерживается")

    return args


if __name__ == "__main__":
    try:
        args = parse_args()

        generator = PanoramaGenerator()

        images = generator.load_images(folder_path=args.folder, image_format=args.image_format)

        if args.mode == "SIFT":
            panorama = generator.make_panorama_SIFT(images=images)
        else:
            panorama = generator.make_panorama_Stitcher(images=images)

        if panorama is not None:
            cv2.imwrite(args.output, panorama)
            print(f"Панорама успешно сохранена как {args.output}!")
        else:
            Exception("Ошибка при создании панорамы.")

    except Exception as err:
        print(f"{err}")
