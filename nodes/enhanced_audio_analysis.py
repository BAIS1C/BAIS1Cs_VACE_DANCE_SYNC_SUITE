"""
enhanced_audio_analysis.py
BAIS1C VACE Dance Sync Suite - Advanced Audio Analysis Engine

Provides the EnhancedAudioAnalyzer class for robust BPM, beat, and frequency analysis
to support dance-video synchronization and music-driven animation pipelines.

Usage:
    from .enhanced_audio_analysis import EnhancedAudioAnalyzer
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Only suppress librosa warnings
warnings.filterwarnings('ignore', module='librosa')

class EnhancedAudioAnalyzer:
    """
    Advanced audio analysis engine for dance synchronization.
    Features:
    - Multi-band frequency analysis (7-band EQ)
    - Robust BPM detection with confidence scoring
    - Beat, bass, and hi-hat pattern recognition
    - Rhythmic consistency validation
    - Onset detection with spectral flux
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.freq_bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'highs': (4000, 8000),
            'air': (8000, 20000)
        }
        self.bpm_range = (60, 180)
        self.confidence_threshold = 0.6

    def analyze_comprehensive(self, waveform: Any, target_fps: float = 24.0, debug: bool = False) -> Dict:
        """
        Comprehensive audio analysis returning all sync-relevant features.
        All frame-based arrays now consistently sized to total_frames.
        """
        # Handle input types robustly
        if hasattr(waveform, 'cpu'):  # torch.Tensor
            waveform = waveform.cpu().numpy()
        if isinstance(waveform, list):
            waveform = np.array(waveform)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        waveform = np.asfarray(waveform)
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))

        duration = len(waveform) / self.sample_rate
        total_frames = int(duration * target_fps)
        if total_frames < 1:
            total_frames = 1

        if debug:
            print(f"[Enhanced Audio] {duration:.2f}s, {self.sample_rate}Hz, {total_frames} frames @ {target_fps} FPS")

        # Pass total_frames to all analysis methods for consistency
        bpm_analysis = self._analyze_bpm_robust(waveform, debug)
        beat_analysis = self._analyze_beat_patterns(waveform, bpm_analysis.get('primary_bpm', 120.0), total_frames, debug)
        freq_analysis = self._analyze_frequency_bands(waveform, total_frames, debug)
        onset_analysis = self._analyze_onsets(waveform, total_frames, debug)
        rhythm_analysis = self._analyze_rhythmic_patterns(waveform, bpm_analysis.get('primary_bpm', 120.0), debug)

        # Ensure all frame-based arrays are consistent
        result = {
            'primary_bpm': bpm_analysis.get('primary_bpm', 120.0),
            'bpm_confidence': bpm_analysis.get('confidence', 0.0),
            'bpm_candidates': bpm_analysis.get('candidates', []),
            'tempo_stability': bpm_analysis.get('stability', 0.0),

            'beat_times': beat_analysis.get('beat_times', []),
            'beat_strength': self._ensure_frame_length(beat_analysis.get('beat_strength', []), total_frames),
            'beat_consistency': beat_analysis.get('consistency', 0.0),

            'freq_bands': freq_analysis,
            'onsets': onset_analysis,
            'rhythm_patterns': rhythm_analysis,

            'total_frames': total_frames,
            'duration': duration,
            'sample_rate': self.sample_rate,

            # Legacy compatibility with consistent sizing
            'beat': self._ensure_frame_length(beat_analysis.get('beat_strength', []), total_frames),
            'bass': self._ensure_frame_length(freq_analysis.get('bass', {}).get('energy', []), total_frames),
            'energy': self._ensure_frame_length(freq_analysis.get('mid', {}).get('energy', []), total_frames)
        }

        if debug:
            self._print_analysis_summary(result)

        return result

    def _ensure_frame_length(self, arr, target_frames):
        """Ensure array matches target frame count exactly"""
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        
        if len(arr) == 0:
            return np.zeros(target_frames)
        
        if len(arr) == target_frames:
            return arr
        
        # Resample to target length
        old_indices = np.linspace(0, len(arr) - 1, len(arr))
        new_indices = np.linspace(0, len(arr) - 1, target_frames)
        return np.interp(new_indices, old_indices, arr)

    def _analyze_bpm_robust(self, waveform: np.ndarray, debug: bool = False) -> Dict:
        """Robust BPM detection with multiple methods and confidence scoring."""
        try:
            # Main librosa tempo
            tempo_global = librosa.beat.tempo(
                y=waveform, sr=self.sample_rate, aggregate=np.median, start_bpm=120, std_bpm=40
            )[0]
            tempo_local = librosa.beat.tempo(
                y=waveform, sr=self.sample_rate, aggregate=None, start_bpm=120, std_bpm=40
            )

            hop_lengths = [512, 1024, 2048]
            tempo_candidates = []
            for hop_length in hop_lengths:
                try:
                    tempo, _ = librosa.beat.beat_track(
                        y=waveform, sr=self.sample_rate, hop_length=hop_length, start_bpm=120
                    )
                    if 60 <= tempo <= 180:
                        tempo_candidates.append(tempo)
                except Exception:
                    continue

            # Onset-based estimate
            try:
                onset_frames = librosa.onset.onset_detect(
                    y=waveform, sr=self.sample_rate, units='time', backtrack=True
                )
                if len(onset_frames) > 10:
                    onset_intervals = np.diff(onset_frames)
                    interval_bpms = 60.0 / onset_intervals
                    valid_bpms = interval_bpms[(interval_bpms >= 60) & (interval_bpms <= 180)]
                    if len(valid_bpms) > 0:
                        tempo_candidates.append(np.median(valid_bpms))
            except Exception:
                pass

            all_estimates = [tempo_global] + list(tempo_local) + tempo_candidates
            all_estimates = [t for t in all_estimates if 60 <= t <= 180]

            if not all_estimates:
                return {'primary_bpm': 120.0, 'confidence': 0.0, 'candidates': []}

            estimates_array = np.array(all_estimates)
            clusters = self._cluster_bpm_candidates(estimates_array, tolerance=0.05)
            best_cluster = max(clusters, key=lambda c: len(c['members']) * c['consistency'])
            primary_bpm = float(self._apply_musical_intelligence(best_cluster['center'], estimates_array))

            confidence = min(1.0, (len(best_cluster['members']) / len(all_estimates)) * best_cluster['consistency'])
            stability = 1.0 - (np.std(estimates_array) / np.mean(estimates_array))
            stability = max(0.0, min(1.0, stability))

            if debug:
                print(f"[BPM Analysis] {primary_bpm:.2f} BPM, conf {confidence:.2f}, stab {stability:.2f}")

            return {
                'primary_bpm': primary_bpm,
                'confidence': confidence,
                'stability': stability,
                'candidates': [{'bpm': float(c['center']), 'weight': len(c['members'])} for c in clusters]
            }
        except Exception as e:
            if debug:
                print(f"[BPM Analysis] Error: {e}")
            return {'primary_bpm': 120.0, 'confidence': 0.0, 'candidates': []}

    def _cluster_bpm_candidates(self, estimates: np.ndarray, tolerance: float = 0.05) -> List[Dict]:
        """Cluster BPM estimates by similarity."""
        if len(estimates) == 0:
            return []
        clusters = []
        used = np.zeros(len(estimates), dtype=bool)
        for i, bpm in enumerate(estimates):
            if used[i]:
                continue
            cluster_mask = np.abs(estimates - bpm) <= (bpm * tolerance)
            cluster_members = estimates[cluster_mask]
            used[cluster_mask] = True
            clusters.append({
                'center': np.mean(cluster_members),
                'members': cluster_members,
                'consistency': 1.0 - (np.std(cluster_members) / np.mean(cluster_members)) if np.mean(cluster_members) > 0 else 0
            })
        return sorted(clusters, key=lambda c: len(c['members']), reverse=True)

    def _apply_musical_intelligence(self, bpm: float, all_estimates: np.ndarray) -> float:
        """Refine BPM estimate for double/half time and snapping to common values."""
        if bpm > 160:
            half_time = bpm / 2
            if 80 <= half_time <= 140:
                half_support = np.sum(np.abs(all_estimates - half_time) <= 5)
                full_support = np.sum(np.abs(all_estimates - bpm) <= 5)
                if half_support >= full_support:
                    bpm = half_time
        elif bpm < 80:
            double_time = bpm * 2
            if 100 <= double_time <= 160:
                double_support = np.sum(np.abs(all_estimates - double_time) <= 5)
                half_support = np.sum(np.abs(all_estimates - bpm) <= 5)
                if double_support > half_support:
                    bpm = double_time
        common_bpms = [90, 100, 110, 120, 128, 130, 140, 150, 160]
        for common in common_bpms:
            if abs(bpm - common) <= 3:
                bpm = common
                break
        return bpm

    def _analyze_beat_patterns(self, waveform: np.ndarray, bpm: float, total_frames: int, debug: bool = False) -> Dict:
        """
        Advanced beat pattern analysis with consistent frame sizing
        """
        try:
            tempo, beat_frames = librosa.beat.beat_track(
                y=waveform, sr=self.sample_rate, units='time', bpm=bpm
            )
            
            beat_times = beat_frames
            duration = len(waveform) / self.sample_rate
            
            # Create beat_strength array with exact target_frames length
            beat_strength = np.zeros(total_frames)

            # Convert beat times to frame indices
            for beat_time in beat_times:
                frame_idx = int((beat_time / duration) * total_frames)
                if 0 <= frame_idx < total_frames:
                    # Apply beat strength in a small window
                    window_start = max(0, frame_idx - 2)
                    window_end = min(total_frames, frame_idx + 3)
                    beat_strength[window_start:window_end] = 1.0

            if len(beat_times) > 1:
                beat_intervals = np.diff(beat_times)
                consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
                consistency = max(0.0, min(1.0, consistency))
            else:
                consistency = 0.0

            sub_beat_times = self._detect_sub_beats(waveform, beat_times, debug)

            result = {
                'beat_times': beat_times,
                'beat_strength': beat_strength,  # Now exactly total_frames length
                'consistency': consistency,
                'sub_beats': sub_beat_times,
                'beat_count': len(beat_times)
            }
            
            if debug:
                print(f"[Beat Analysis] {len(beat_times)} beats, Beat strength shape: {beat_strength.shape}, Consistency: {consistency:.2f}")
            
            return result
        except Exception as e:
            if debug:
                print(f"[Beat Analysis] Error: {e}")
            return {
                'beat_times': np.array([]),
                'beat_strength': np.zeros(total_frames),  # Consistent fallback
                'consistency': 0.0,
                'sub_beats': np.array([]),
                'beat_count': 0
            }

    def _detect_sub_beats(self, waveform: np.ndarray, beat_times: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Detect sub-beat elements like hi-hats and off-beats
        """
        if len(beat_times) < 2:
            return np.array([])
        try:
            high_freq = librosa.effects.preemphasis(waveform)
            onset_times = librosa.onset.onset_detect(
                y=high_freq, sr=self.sample_rate,
                units='time', backtrack=True,
                pre_max=0.03, post_max=0.03, pre_avg=0.1, post_avg=0.1,
                delta=0.1, wait=0.02
            )
            beat_tolerance = 0.05
            sub_beats = []
            for onset in onset_times:
                is_main_beat = np.any(np.abs(beat_times - onset) <= beat_tolerance)
                if not is_main_beat:
                    sub_beats.append(onset)
            return np.array(sub_beats)
        except Exception as e:
            if debug:
                print(f"[Sub-beat Detection] Error: {e}")
            return np.array([])

    def _analyze_frequency_bands(self, waveform: np.ndarray, total_frames: int, debug: bool = False) -> Dict:
        """
        Multi-band EQ analysis with consistent frame sizing
        """
        try:
            stft = librosa.stft(waveform, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
            band_analysis = {}

            for band_name, (low_freq, high_freq) in self.freq_bands.items():
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if not np.any(freq_mask):
                    band_analysis[band_name] = {
                        'energy': np.zeros(total_frames),
                        'peak_freq': low_freq,
                        'spectral_centroid': (low_freq + high_freq) / 2,
                        'rms': 0.0
                    }
                    continue
                
                band_magnitude = magnitude[freq_mask, :]
                band_energy = np.mean(band_magnitude, axis=0)
                
                # Always resample to exact total_frames length
                if len(band_energy) != total_frames:
                    band_energy = self._ensure_frame_length(band_energy, total_frames)
                
                if np.max(band_energy) > 0:
                    band_energy = band_energy / np.max(band_energy)
                
                peak_freq_idx = np.argmax(np.mean(band_magnitude, axis=1))
                if len(freqs[freq_mask]) > 0 and peak_freq_idx < len(freqs[freq_mask]):
                    peak_freq = freqs[freq_mask][peak_freq_idx]
                else:
                    peak_freq = low_freq
                
                spectral_centroid = np.average(
                    freqs[freq_mask], weights=np.mean(band_magnitude, axis=1)
                ) if np.any(freq_mask) else (low_freq + high_freq) / 2
                
                rms_energy = np.sqrt(np.mean(band_energy ** 2))
                
                band_analysis[band_name] = {
                    'energy': band_energy,  # Now guaranteed to be total_frames length
                    'peak_freq': float(peak_freq),
                    'spectral_centroid': float(spectral_centroid),
                    'rms': float(rms_energy)
                }
                
            if debug:
                print(f"[Frequency Analysis] {len(self.freq_bands)} bands analyzed, each with {total_frames} frames")
                
            return band_analysis
            
        except Exception as e:
            if debug:
                print(f"[Frequency Analysis] Error: {e}")
            empty_energy = np.zeros(total_frames)
            return {band: {'energy': empty_energy, 'peak_freq': 0, 'spectral_centroid': 0, 'rms': 0.0}
                    for band in self.freq_bands.keys()}

    def _analyze_onsets(self, waveform: np.ndarray, total_frames: int, debug: bool = False) -> Dict:
        """
        Advanced onset detection with multiple methods
        """
        try:
            methods = ['energy', 'hfc', 'complex', 'phase', 'specdiff', 'kl', 'mkl']
            onset_results = {}
            for method in methods:
                try:
                    if method == 'energy':
                        onset_env = librosa.onset.onset_strength(y=waveform, sr=self.sample_rate)
                    elif method == 'hfc':
                        stft = librosa.stft(waveform)
                        onset_env = np.sum(np.abs(stft) * np.arange(stft.shape[0])[:, np.newaxis], axis=0)
                    else:
                        onset_env = librosa.onset.onset_strength(y=waveform, sr=self.sample_rate,
                                                                 feature=librosa.feature.melspectrogram)
                    onset_times = librosa.onset.onset_detect(
                        onset_envelope=onset_env, sr=self.sample_rate, units='time'
                    )
                    onset_strength = np.zeros(total_frames)
                    for onset_time in onset_times:
                        frame_idx = int((onset_time / (len(waveform) / self.sample_rate)) * total_frames)
                        if 0 <= frame_idx < total_frames:
                            onset_strength[frame_idx] = 1.0
                    onset_results[method] = {
                        'times': onset_times,
                        'strength': onset_strength,
                        'count': len(onset_times)
                    }
                except Exception as method_error:
                    if debug:
                        print(f"[Onset] Method {method} failed: {method_error}")
                    continue
            if onset_results:
                combined_strength = np.zeros(total_frames)
                for method_data in onset_results.values():
                    combined_strength += method_data['strength']
                combined_strength /= len(onset_results)
                onset_threshold = 0.3
                onset_frames = np.where(combined_strength > onset_threshold)[0]
                combined_times = onset_frames * (len(waveform) / self.sample_rate) / total_frames
                result = {
                    'combined_strength': combined_strength,
                    'combined_times': combined_times,
                    'methods': onset_results,
                    'total_onsets': len(combined_times)
                }
            else:
                result = {
                    'combined_strength': np.zeros(total_frames),
                    'combined_times': np.array([]),
                    'methods': {},
                    'total_onsets': 0
                }
            if debug:
                print(f"[Onset Analysis] {result['total_onsets']} combined onsets")
                for method, data in onset_results.items():
                    print(f"  {method}: {data['count']} onsets")
            return result
        except Exception as e:
            if debug:
                print(f"[Onset Analysis] Error: {e}")
            return {
                'combined_strength': np.zeros(total_frames),
                'combined_times': np.array([]),
                'methods': {},
                'total_onsets': 0
            }

    def _analyze_rhythmic_patterns(self, waveform: np.ndarray, bpm: float, debug: bool = False) -> Dict:
        """
        Analyze rhythmic patterns and groove characteristics
        """
        try:
            beat_period = 60.0 / bpm
            tempo, beat_times = librosa.beat.beat_track(
                y=waveform, sr=self.sample_rate, units='time', bpm=bpm
            )
            rhythm_features = {}
            if len(beat_times) > 4:
                beat_intervals = np.diff(beat_times)
                rhythm_features['tempo_consistency'] = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
                rhythm_features['average_interval'] = float(np.mean(beat_intervals))
                if len(beat_times) > 8:
                    odd_intervals = beat_intervals[::2]
                    even_intervals = beat_intervals[1::2]
                    if len(odd_intervals) > 0 and len(even_intervals) > 0:
                        swing_ratio = np.mean(odd_intervals) / np.mean(even_intervals)
                        rhythm_features['swing_ratio'] = float(swing_ratio)
                        rhythm_features['has_swing'] = abs(swing_ratio - 1.0) > 0.1
                    else:
                        rhythm_features['swing_ratio'] = 1.0
                        rhythm_features['has_swing'] = False
                else:
                    rhythm_features['swing_ratio'] = 1.0
                    rhythm_features['has_swing'] = False
                onset_times = librosa.onset.onset_detect(
                    y=waveform, sr=self.sample_rate, units='time'
                )
                syncopation_score = 0.0
                if len(onset_times) > 0:
                    syncopated_onsets = 0
                    for onset in onset_times:
                        beat_distances = np.abs(beat_times - onset)
                        min_distance = np.min(beat_distances)
                        if min_distance > beat_period * 0.2:
                            syncopated_onsets += 1
                    syncopation_score = syncopated_onsets / len(onset_times)
                rhythm_features['syncopation'] = float(syncopation_score)
            else:
                rhythm_features = {
                    'tempo_consistency': 0.0,
                    'average_interval': beat_period,
                    'swing_ratio': 1.0,
                    'has_swing': False,
                    'syncopation': 0.0
                }
            rhythm_features['groove_strength'] = self._calculate_groove_strength(waveform, beat_times)
            if debug:
                print(f"[Rhythm Analysis] Patterns: Tempo consistency: {rhythm_features['tempo_consistency']:.2f}, "
                      f"Swing: {'Yes' if rhythm_features['has_swing'] else 'No'} ({rhythm_features['swing_ratio']:.2f}), "
                      f"Syncopation: {rhythm_features['syncopation']:.2f}, "
                      f"Groove: {rhythm_features['groove_strength']:.2f}")
            return rhythm_features
        except Exception as e:
            if debug:
                print(f"[Rhythm Analysis] Error: {e}")
            return {
                'tempo_consistency': 0.0,
                'average_interval': 60.0 / bpm,
                'swing_ratio': 1.0,
                'has_swing': False,
                'syncopation': 0.0,
                'groove_strength': 0.0
            }

    def _calculate_groove_strength(self, waveform: np.ndarray, beat_times: np.ndarray) -> float:
        """
        Calculate the "groove strength" - how locked in the rhythm is
        """
        try:
            if len(beat_times) < 4:
                return 0.0
            beat_energy = []
            window_size = int(0.1 * self.sample_rate)
            for beat_time in beat_times:
                beat_sample = int(beat_time * self.sample_rate)
                start_idx = max(0, beat_sample - window_size // 2)
                end_idx = min(len(waveform), beat_sample + window_size // 2)
                if end_idx > start_idx:
                    window_energy = np.mean(waveform[start_idx:end_idx] ** 2)
                    beat_energy.append(window_energy)
            off_beat_energy = []
            for i in range(len(beat_times) - 1):
                off_beat_time = (beat_times[i] + beat_times[i + 1]) / 2
                off_beat_sample = int(off_beat_time * self.sample_rate)
                start_idx = max(0, off_beat_sample - window_size // 2)
                end_idx = min(len(waveform), off_beat_sample + window_size // 2)
                if end_idx > start_idx:
                    window_energy = np.mean(waveform[start_idx:end_idx] ** 2)
                    off_beat_energy.append(window_energy)
            if len(beat_energy) > 0 and len(off_beat_energy) > 0:
                avg_beat_energy = np.mean(beat_energy)
                avg_off_beat_energy = np.mean(off_beat_energy)
                if avg_off_beat_energy > 0:
                    groove_ratio = avg_beat_energy / (avg_beat_energy + avg_off_beat_energy)
                    return float(groove_ratio)
            return 0.5
        except Exception:
            return 0.0

    def _print_analysis_summary(self, analysis: Dict):
        print(f"\n[Enhanced Audio Analysis] === SUMMARY ===")
        print(f"BPM: {analysis['primary_bpm']:.1f} (confidence: {analysis['bpm_confidence']:.2f})")
        print(f"Duration: {analysis['duration']:.1f}s ({analysis['total_frames']} frames)")
        print(f"Beat consistency: {analysis['beat_consistency']:.2f}")
        print(f"\nFrequency Bands:")
        for band_name, band_data in analysis['freq_bands'].items():
            print(f"  {band_name}: RMS={band_data['rms']:.3f}, Peak={band_data['peak_freq']:.0f}Hz")
        print(f"\nRhythmic Patterns:")
        rhythm = analysis['rhythm_patterns']
        print(f"  Tempo consistency: {rhythm['tempo_consistency']:.2f}")
        print(f"  Swing: {'Yes' if rhythm['has_swing'] else 'No'} (ratio: {rhythm['swing_ratio']:.2f})")
        print(f"  Syncopation: {rhythm['syncopation']:.2f}")
        print(f"  Groove strength: {rhythm['groove_strength']:.2f}")
        print(f"\nOnsets: {analysis['onsets']['total_onsets']} detected")
        print(f"Beat tracking: {len(analysis['beat_times'])} beats")

# (No top-level code; safe for import.)