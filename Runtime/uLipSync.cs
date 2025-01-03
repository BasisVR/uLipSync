using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using System.Collections.Generic;

namespace uLipSync
{

    public class uLipSync : MonoBehaviour
    {
        public Profile profile;
        JobHandle _jobHandle;
        object _lockObject = new object();
        bool _allocated = false;
        int _index = 0;
        bool _isDataReceived = false;
        NativeArray<float> _inputData;
        NativeArray<float> _mfcc;
        NativeArray<float> _mfccForOther;
        NativeArray<float> _means;
        NativeArray<float> _standardDeviations;
        NativeArray<float> _phonemes;
        NativeArray<float> _scores;
        NativeArray<LipSyncJob.Info> _info;
        List<int> _requestedCalibrationVowels = new List<int>();
        Dictionary<string, float> _ratios = new Dictionary<string, float>();
        public float[] UpdateResultsBuffer;
        public int phonemeCount;
        public NativeArray<float> mfcc => _mfccForOther;
        public LipSyncInfo result { get; private set; } = new LipSyncInfo();
        public float[] Inputs;
        int mfccNum => profile ? profile.mfccNum : 12;
        public uLipSyncBlendShape uLipSyncBlendShape;
        public void Initalize()
        {
            AllocateBuffers();
        }

        void OnDisable()
        {
            _jobHandle.Complete();
            DisposeBuffers();
        }

        public void LateUpdate()
        {
            if (_jobHandle.IsCompleted == false)
            {
                return;
            }
            _jobHandle.Complete(); // Wait for async job completion
            _mfccForOther.CopyFrom(_mfcc);

            // Main phoneme identification
            int index = _info[0].mainPhonemeIndex;
            string mainPhoneme = profile.GetPhoneme(index);

            // Calculate sumScore and populate UpdateResultsBuffer
            float sumScore = 0f;
            _scores.CopyTo(UpdateResultsBuffer);
            for (int i = 0; i < phonemeCount; ++i)
            {
                sumScore += UpdateResultsBuffer[i];
            }

            // Clear and update _ratios
            _ratios.Clear();
            float invSumScore = sumScore > 0f ? 1f / sumScore : 0f; // Precompute inverse for efficiency
            for (int i = 0; i < phonemeCount; ++i)
            {
                string phoneme = profile.GetPhoneme(i);
                float ratio = UpdateResultsBuffer[i] * invSumScore; // Avoid division in loop
                if (!_ratios.TryGetValue(phoneme, out float existingRatio))
                {
                    _ratios[phoneme] = ratio; // Add new
                }
                else
                {
                    _ratios[phoneme] = existingRatio + ratio; // Accumulate
                }
            }

            // Normalize volume
            float rawVol = _info[0].volume;
            float normVol = math.clamp((math.log10(rawVol) - Common.DefaultMinVolume) / (Common.DefaultMaxVolume - Common.DefaultMinVolume), 0f, 1f);

            // Update result
            result = new LipSyncInfo()
            {
                phoneme = mainPhoneme,
                volume = normVol,
                rawVolume = rawVol,
                phonemeRatios = _ratios,
            };
            index = 0;
            uLipSyncBlendShape.OnLipSyncUpdate(result);
            int Count = _requestedCalibrationVowels.Count;
            for (int Index = 0; Index < Count; Index++)
            {
                index = _requestedCalibrationVowels[Index];
                profile.UpdateMfcc(index, mfcc, true);
            }

            _requestedCalibrationVowels.Clear();
             index = 0;

            for (int i = 0; i < mfccsCount && index < PhonemesCount; i++)
            {
                NativeArray<float> mfccNativeArray = profile.mfccs[i].mfccNativeArray;

                // Determine how many elements to copy
                int remainingLength = PhonemesCount - index;
                int copyLength = math.min(12, remainingLength);

                // Use NativeArray.CopyTo for batch copying
                NativeArray<float>.Copy(mfccNativeArray, 0, _phonemes, index, copyLength);

                index += copyLength;
            }
            if (!_isDataReceived) return;
            _isDataReceived = false;

            CachedInputSampleCount = inputSampleCount;
            index = 0;
            lock (_lockObject)
            {
                _inputData.CopyFrom(Inputs);
                index = _index;
            }

            LipSyncJob lipSyncJob = new LipSyncJob()
            {
                input = _inputData,
                startIndex = index,
                outputSampleRate = outputSampleRate,
                targetSampleRate = profile.targetSampleRate,
                melFilterBankChannels = profile.melFilterBankChannels,
                means = _means,
                standardDeviations = _standardDeviations,
                mfcc = _mfcc,
                phonemes = _phonemes,
                compareMethod = profile.compareMethod,
                scores = _scores,
                info = _info,
            };

            _jobHandle = lipSyncJob.Schedule();
        }
        void AllocateBuffers()
        {
            if (_allocated)
            {
                DisposeBuffers();
            }
            _allocated = true;

            _jobHandle.Complete();

            lock (_lockObject)
            {
                CachedInputSampleCount = inputSampleCount;
                phonemeCount = profile ? profile.mfccs.Count : 1;
                Inputs = new float[CachedInputSampleCount];
                _inputData = new NativeArray<float>(CachedInputSampleCount, Allocator.Persistent);
                _mfcc = new NativeArray<float>(mfccNum, Allocator.Persistent);
                _mfccForOther = new NativeArray<float>(mfccNum, Allocator.Persistent);
                _means = new NativeArray<float>(mfccNum, Allocator.Persistent);
                _standardDeviations = new NativeArray<float>(mfccNum, Allocator.Persistent);
                _scores = new NativeArray<float>(phonemeCount, Allocator.Persistent);
                UpdateResultsBuffer = new float[phonemeCount];
                _phonemes = new NativeArray<float>(mfccNum * phonemeCount, Allocator.Persistent);
                _info = new NativeArray<LipSyncJob.Info>(1, Allocator.Persistent);
                _means.CopyFrom(profile.means);
                _standardDeviations.CopyFrom(profile.standardDeviation);
                PhonemesCount = mfccNum * phonemeCount;
                mfccsCount = profile.mfccs.Count;
                outputSampleRate = AudioSettings.outputSampleRate;
            }
        }
        public int outputSampleRate;
        public int PhonemesCount;
        public int mfccsCount;
        void DisposeBuffers()
        {
            if (!_allocated) return;
            _allocated = false;

            _jobHandle.Complete();

            lock (_lockObject)
            {
                Inputs = null;
                _inputData.Dispose();
                _mfcc.Dispose();
                _mfccForOther.Dispose();
                _means.Dispose();
                _standardDeviations.Dispose();
                _scores.Dispose();
                _phonemes.Dispose();
                _info.Dispose();
            }
        }
        public void RequestCalibration(int index)
        {
            _requestedCalibrationVowels.Add(index);
        }
        int inputSampleCount
        {
            get
            {
                if (!profile) return AudioSettings.outputSampleRate;
                float r = (float)AudioSettings.outputSampleRate / profile.targetSampleRate;
                return Mathf.CeilToInt(profile.sampleCount * r);
            }
        }
        public int CachedInputSampleCount;
        public void OnDataReceived(float[] input, int channels,int length)
        {
            lock (_lockObject)
            {
                _index = _index % CachedInputSampleCount;
                for (int i = 0; i < length; i += channels)
                {
                    Inputs[_index++ % CachedInputSampleCount] = input[i];
                }
            }

            _isDataReceived = true;
        }
    }

}
