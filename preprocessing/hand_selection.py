"""Select one WiLoR detection using the supplied hand mask."""
import logging
from pathlib import Path
import numpy as np

class HandSelector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _detect_data_format(self, array):
            'Identify regular or object-array WiLoR output and its frame and hand counts.'

            if array.ndim == 4:  # (n_frames, n_hands, n_verts/n_joints, 3/2)

                if array.shape[2] in [778, 21]:  # 778 vertices or 21 joints
                    return 'regular_multi', array.shape[0], array.shape[1]

            if array.ndim == 3:
                return 'regular_single', array.shape[0], 1


            if array.dtype == object and array.ndim <= 2:
                n_frames = len(array)
                hand_counts = [len(frame_data) if isinstance(frame_data, list) else 1
                              for frame_data in array]
                max_hands = max(hand_counts)

                if max_hands <= 1:
                    return 'object_single', n_frames, 1
                else:
                    return 'object_list', n_frames, max_hands


            if array.ndim >= 2:

                if array.shape[1] > 1:
                    return 'regular_multi', array.shape[0], array.shape[1]


            return 'regular_single', array.shape[0], 1

    def _normalize_to_array(self, data, format_type, n_frames, n_hands):
            'Normalize WiLoR data to a numeric array with shape (frames, hands, ...).'
            if format_type == 'object_list':
                # Convert list format to array format
                # Find the shape of hand data from first non-empty frame
                template_shape = None
                for frame_data in data:
                    if isinstance(frame_data, list) and len(frame_data) > 0:
                        template_shape = frame_data[0].shape
                        break

                if template_shape is None:
                    raise ValueError("Cannot determine hand data shape from object array")

                result_list = []
                for frame_data in data:
                    if isinstance(frame_data, list):
                        # Pad to n_hands if needed
                        frame_array = []
                        for i in range(n_hands):
                            if i < len(frame_data):
                                frame_array.append(frame_data[i])
                            else:
                                # Pad with zeros of same shape
                                frame_array.append(np.zeros(template_shape, dtype=np.float32))
                        result_list.append(np.array(frame_array))
                    else:
                        # Single hand, wrap in array
                        padded = [frame_data] + [np.zeros(template_shape, dtype=np.float32) for _ in range(n_hands - 1)]
                        result_list.append(np.array(padded))

                return np.array(result_list)

            elif format_type == 'regular_multi':

                if data.dtype == object:
                    return np.array(data, dtype=np.float32)
                return data

            elif format_type in ['regular_single', 'object_single']:
                # Add hand dimension: (n_frames, ...) -> (n_frames, 1, ...)
                if data.dtype == object:
                    # Need to convert object array to regular array first
                    result_list = []
                    for frame_data in data:
                        if isinstance(frame_data, list):
                            result_list.append(frame_data[0] if len(frame_data) > 0 else frame_data)
                        else:
                            result_list.append(frame_data)
                    data = np.array(result_list)
                return np.expand_dims(data, axis=1)

    def _filter_and_reshape(self, array, format_type, selected_indices, n_frames):
            'Select the requested hand in each frame and retain a singleton hand axis.'
            if format_type == 'object_list':
                # Extract selected hand from each frame's list
                result = []
                for i in range(n_frames):
                    frame_data = array[i]
                    if isinstance(frame_data, list):
                        hand_idx = selected_indices[i]
                        if hand_idx < len(frame_data):
                            result.append(frame_data[hand_idx])
                        else:
                            # Fallback to first hand if index out of range
                            self.logger.warning(f"Frame {i}: selected hand index {hand_idx} out of range, using hand 0")
                            result.append(frame_data[0] if len(frame_data) > 0 else frame_data)
                    else:
                        result.append(frame_data)

                # Convert to array and add hand dimension
                result_array = np.array(result)
                return np.expand_dims(result_array, axis=1)

            elif format_type == 'regular_multi':
                # Use existing logic
                filtered = np.array([array[i, selected_indices[i]] for i in range(n_frames)])
                return np.expand_dims(filtered, axis=1)

            else:
                # Already single hand, just ensure shape (n_frames, 1, ...)
                if array.dtype == object:
                    # Convert object array to regular array first
                    result_list = []
                    for frame_data in array:
                        if isinstance(frame_data, list):
                            result_list.append(frame_data[0] if len(frame_data) > 0 else frame_data)
                        else:
                            result_list.append(frame_data)
                    array = np.array(result_list)

                if array.ndim >= 2 and array.shape[1] == 1:
                    return array
                else:
                    return np.expand_dims(array, axis=1)

    def filter_wilor_by_hand_mask(self, wilor_output_dir: str, mask_base_dir: str):
            'Select the WiLoR hand that overlaps the supplied hand mask.'
            print(f"Filtering WiLoR output by hand mask...")

            from preprocessing.single_hand import single_hand_mask_directory
            hand_mask_dir = single_hand_mask_directory(mask_base_dir)
            return self._filter_wilor_single_hand(wilor_output_dir, str(hand_mask_dir))

    def _filter_wilor_single_hand(self, wilor_output_dir: str, hand_mask_dir: str):
            'Filter WiLoR output to one hand per frame.'

            verts_file = Path(wilor_output_dir) / "mano_verts.npz"
            joints_2d_file = Path(wilor_output_dir) / "mano_joints_2d.npz"

            if not verts_file.exists():
                print(f"Warning: mano_verts.npz not found in {wilor_output_dir}")
                return False


            verts_data = np.load(verts_file, allow_pickle=True)
            verts = verts_data['mano_verts']


            format_type, n_frames, n_hands = self._detect_data_format(verts)


            print(f"\n{'='*60}")
            print(f"WiLoR Hand Detection Summary (Single Hand)")
            print(f"{'='*60}")
            print(f"Total frames: {n_frames}")
            print(f"Number of hands detected: {n_hands}")
            print(f"Data format: {format_type}")
            print(f"Verts dtype: {verts.dtype}")
            print(f"Verts shape: {verts.shape}")


            if n_hands == 1:

                print(f"{'='*60}\n")

                self.logger.info(f"Detected {n_hands} hand in {n_frames} frames")
                self.logger.info(f"Single hand detection, ensuring output format is (n_frames, 1, ...)")


                npz_files = {
                    'mano_verts.npz': 'mano_verts',
                    'mano_joints.npz': 'mano_joints',
                    'mano_joints_2d.npz': 'mano_joints_2d',
                    'mano_global_orient.npz': 'mano_global_orient',
                    'mano_hand_pose.npz': 'mano_hand_pose',
                    'mano_betas.npz': 'mano_betas',
                }

                for filename, key_name in npz_files.items():
                    npz_path = Path(wilor_output_dir) / filename
                    if not npz_path.exists():
                        continue

                    data = np.load(npz_path, allow_pickle=True)
                    if key_name not in data:
                        continue

                    array = data[key_name]
                    file_format_type, _, _ = self._detect_data_format(array)


                    if file_format_type in ['regular_single', 'object_single']:

                        reshaped_array = self._filter_and_reshape(array, file_format_type, [0] * n_frames, n_frames)
                        self.logger.info(f"Reshaped {filename}: {array.shape} -> {reshaped_array.shape}")
                        np.savez(npz_path, **{key_name: reshaped_array})
                    elif file_format_type in ['regular_multi', 'object_list'] and array.shape[1] == 1:

                        self.logger.info(f"Skipping {filename}: already in correct format {array.shape}")
                    else:
                        self.logger.warning(f"Unexpected format for {filename}: {array.shape}, {file_format_type}")

                return True


            self.logger.info(f"Detected {n_hands} hand(s) in {n_frames} frames")
            self.logger.info(f"Multi-hand detection ({n_hands} hands), filtering by hand mask...")


            if format_type in ['object_list', 'object_single']:
                verts_normalized = self._normalize_to_array(verts, format_type, n_frames, n_hands)
            else:
                verts_normalized = verts


            if format_type in ['regular_multi', 'object_list']:
                if verts_normalized.dtype != object and verts_normalized.ndim >= 4:
                    print(f"Verts data range: [{verts_normalized.min():.3f}, {verts_normalized.max():.3f}]")


                    for hand_idx in range(n_hands):
                        hand_verts = verts_normalized[:, hand_idx, :, :]  # (n_frames, n_verts, 3)
                        mean_pos = hand_verts.mean(axis=(0, 1))
                        print(f"\nHand {hand_idx}:")
                        print(f"  Mean position: [{mean_pos[0]:.3f}, {mean_pos[1]:.3f}, {mean_pos[2]:.3f}]")
                        print(f"  Position range X: [{hand_verts[:,:,0].min():.3f}, {hand_verts[:,:,0].max():.3f}]")
                        print(f"  Position range Y: [{hand_verts[:,:,1].min():.3f}, {hand_verts[:,:,1].max():.3f}]")
                        print(f"  Position range Z: [{hand_verts[:,:,2].min():.3f}, {hand_verts[:,:,2].max():.3f}]")


            if joints_2d_file.exists():
                joints_2d_data = np.load(joints_2d_file, allow_pickle=True)
                joints_2d_raw = joints_2d_data['mano_joints_2d']


                joints_2d_format_type, _, _ = self._detect_data_format(joints_2d_raw)
                joints_2d_normalized = self._normalize_to_array(joints_2d_raw, joints_2d_format_type, n_frames, n_hands)

                print(f"\n2D Joints shape: {joints_2d_normalized.shape}")
                if joints_2d_normalized.dtype != object and joints_2d_normalized.ndim >= 3:
                    for hand_idx in range(n_hands):
                        hand_joints = joints_2d_normalized[:, hand_idx, :, :]  # (n_frames, n_joints, 2)

                        valid_count = (~np.isnan(hand_joints)).all(axis=2).sum()
                        total_count = n_frames * joints_2d_normalized.shape[2]
                        print(f"  Hand {hand_idx} valid joints: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")


                        valid_joints = hand_joints[~np.isnan(hand_joints).any(axis=2)]
                        if len(valid_joints) > 0:
                            print(f"  Hand {hand_idx} 2D range X: [{valid_joints[:,0].min():.1f}, {valid_joints[:,0].max():.1f}]")
                            print(f"  Hand {hand_idx} 2D range Y: [{valid_joints[:,1].min():.1f}, {valid_joints[:,1].max():.1f}]")

            print(f"{'='*60}\n")


            if not joints_2d_file.exists():
                self.logger.error("mano_joints_2d.npz not found, cannot perform filtering")
                return False

            joints_2d_data = np.load(joints_2d_file, allow_pickle=True)
            joints_2d_raw = joints_2d_data['mano_joints_2d']


            joints_2d_format_type, _, _ = self._detect_data_format(joints_2d_raw)
            joints_2d = self._normalize_to_array(joints_2d_raw, joints_2d_format_type, n_frames, n_hands)


            selected_hand_indices = self._select_hands_by_mask(joints_2d, hand_mask_dir)


            npz_files = {
                'mano_verts.npz': 'mano_verts',
                'mano_joints.npz': 'mano_joints',
                'mano_joints_2d.npz': 'mano_joints_2d',
                'mano_global_orient.npz': 'mano_global_orient',
                'mano_hand_pose.npz': 'mano_hand_pose',
                'mano_betas.npz': 'mano_betas',
            }

            for filename, key_name in npz_files.items():
                npz_path = Path(wilor_output_dir) / filename
                if not npz_path.exists():
                    continue

                data = np.load(npz_path, allow_pickle=True)
                if key_name not in data:
                    continue

                array = data[key_name]


                file_format_type, _, _ = self._detect_data_format(array)


                filtered_array = self._filter_and_reshape(array, file_format_type, selected_hand_indices, n_frames)
                self.logger.info(f"Filtered {filename}: {array.shape} (dtype={array.dtype}) -> {filtered_array.shape} (dtype={filtered_array.dtype})")


                np.savez(npz_path, **{key_name: filtered_array})

            return True

    def _select_hands_by_mask(self, joints_2d: np.ndarray, hand_mask_dir: str) -> list:
            'Choose a hand per frame using the IoU of its projected joint hull and the supplied mask.'

            import re
            mask_files = list(Path(hand_mask_dir).glob("*.png"))
            if not mask_files:
                mask_files = list(Path(hand_mask_dir).glob("*.jpg"))

            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

            mask_files = sorted(mask_files, key=natural_sort_key)

            n_frames, n_hands = joints_2d.shape[0], joints_2d.shape[1]


            vis_dir = f"{self.config.video_path}/wilor_filtering_vis"
            os.makedirs(vis_dir, exist_ok=True)
            os.makedirs(f"{vis_dir}/before", exist_ok=True)
            os.makedirs(f"{vis_dir}/after", exist_ok=True)


            selected_hand_indices = []


            rgb_dir = f"{self.config.video_path}/{self.config.rgb_subdir}"
            rgb_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                rgb_files.extend(Path(rgb_dir).glob(ext))
            rgb_files = sorted(rgb_files, key=natural_sort_key)

            print(f"Found {len(rgb_files)} RGB files for visualization")
            self.logger.info(f"RGB files: {len(rgb_files)}, Mask files: {len(mask_files)}")

            for frame_idx in range(min(n_frames, len(mask_files))):

                mask_path = mask_files[frame_idx]
                gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if gt_mask is None:
                    self.logger.warning(f"Failed to load mask: {mask_path}")
                    selected_hand_indices.append(0)
                    continue


                rgb_img = None
                if frame_idx < len(rgb_files):
                    rgb_img = cv2.imread(str(rgb_files[frame_idx]))
                    if rgb_img is None:
                        self.logger.warning(f"Failed to load RGB image: {rgb_files[frame_idx]}")
                    elif frame_idx == 0:
                        self.logger.info(f"Successfully loaded RGB image: {rgb_files[frame_idx]}, shape: {rgb_img.shape}")


                gt_mask = (gt_mask > 127).astype(np.uint8)


                best_iou = 0
                best_hand_idx = 0
                hand_ious = []

                for hand_idx in range(n_hands):

                    hand_joints = joints_2d[frame_idx, hand_idx]  # (n_joints, 2)


                    hand_mask = np.zeros_like(gt_mask)


                    valid_joints = hand_joints[~np.isnan(hand_joints).any(axis=1)]
                    if len(valid_joints) > 0:
                        pts = valid_joints.astype(np.int32)

                        cv2.fillConvexPoly(hand_mask, pts, 1)


                        intersection = np.logical_and(hand_mask, gt_mask).sum()
                        union = np.logical_or(hand_mask, gt_mask).sum()

                        if union > 0:
                            iou = intersection / union
                            hand_ious.append(iou)
                            if iou > best_iou:
                                best_iou = iou
                                best_hand_idx = hand_idx
                        else:
                            hand_ious.append(0.0)
                    else:
                        hand_ious.append(0.0)

                selected_hand_indices.append(best_hand_idx)


                if rgb_img is not None and frame_idx % 10 == 0:
                    self.logger.info(f"Attempting to save visualization for frame {frame_idx}")
                    try:
                        self._visualize_hands_before(rgb_img, joints_2d[frame_idx], hand_ious,
                                                     frame_idx, vis_dir, gt_mask)
                        self._visualize_hands_after(rgb_img, joints_2d[frame_idx], best_hand_idx,
                                                    best_iou, frame_idx, vis_dir, gt_mask)
                        self.logger.info(f"Successfully saved visualization for frame {frame_idx}")
                    except Exception as e:
                        self.logger.error(f"Failed to save visualization for frame {frame_idx}: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())

                if best_iou > 0:
                    self.logger.info(f"Frame {frame_idx}: selected hand {best_hand_idx} with IoU={best_iou:.3f}")
                else:
                    self.logger.warning(f"Frame {frame_idx}: no valid IoU, defaulting to hand {best_hand_idx}")


            while len(selected_hand_indices) < n_frames:
                selected_hand_indices.append(selected_hand_indices[-1] if selected_hand_indices else 0)

            print(f"\nVisualization saved to: {vis_dir}")
            return selected_hand_indices

    def _visualize_hands_before(self, rgb_img, joints_2d_frame, hand_ious, frame_idx, vis_dir, gt_mask):
            'Render projected joints for every detected hand before selection.'
            vis_img = rgb_img.copy()
            n_hands = joints_2d_frame.shape[0]


            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]


            mask_overlay = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)
            mask_overlay[:, :, 1] = 0
            mask_overlay[:, :, 0] = 0
            vis_img = cv2.addWeighted(vis_img, 0.7, mask_overlay, 0.3, 0)

            for hand_idx in range(n_hands):
                hand_joints = joints_2d_frame[hand_idx]  # (n_joints, 2)
                color = colors[hand_idx % len(colors)]


                valid_joints = hand_joints[~np.isnan(hand_joints).any(axis=1)]
                for joint in valid_joints:
                    cv2.circle(vis_img, (int(joint[0]), int(joint[1])), 5, color, -1)


                if len(valid_joints) > 0:
                    text_pos = (int(valid_joints[0][0]) + 15, int(valid_joints[0][1]) - 15)
                    iou_text = f"Hand {hand_idx}: IoU={hand_ious[hand_idx]:.3f}"
                    cv2.putText(vis_img, iou_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, color, 2)


            save_path = f"{vis_dir}/before/frame_{frame_idx:04d}_before.jpg"
            cv2.imwrite(save_path, vis_img)

    def _visualize_hands_after(self, rgb_img, joints_2d_frame, selected_hand_idx, iou, frame_idx, vis_dir, gt_mask):
            'Render projected joints of the selected hand.'
            vis_img = rgb_img.copy()


            mask_overlay = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)
            mask_overlay[:, :, 1] = 0
            mask_overlay[:, :, 0] = 0
            vis_img = cv2.addWeighted(vis_img, 0.7, mask_overlay, 0.3, 0)


            hand_joints = joints_2d_frame[selected_hand_idx]
            color = (0, 255, 0)


            valid_joints = hand_joints[~np.isnan(hand_joints).any(axis=1)]
            for joint in valid_joints:
                cv2.circle(vis_img, (int(joint[0]), int(joint[1])), 6, color, -1)


            if len(valid_joints) > 0:
                text_pos = (int(valid_joints[0][0]) + 15, int(valid_joints[0][1]) - 15)
                iou_text = f"Selected Hand {selected_hand_idx}: IoU={iou:.3f}"
                cv2.putText(vis_img, iou_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, color, 2)


            save_path = f"{vis_dir}/after/frame_{frame_idx:04d}_after.jpg"
            cv2.imwrite(save_path, vis_img)
