import numpy as np
import mindpark as mp
import mindpark.part.replay


class TestRingBuffer:

    def test_length(self):
        memory = mp.part.replay.RingBuffer(5, [[]])
        for number in range(10):
            assert len(memory) == min(number, 5)
            memory.push(number)

    def test_override_last_elements(self):
        memory = mp.part.replay.RingBuffer(5, [[]])
        for number in range(10):
            memory.push(number)
        assert (memory[:][0] == [5, 6, 7, 8, 9]).all()

    def test_slicing_underfull(self):
        memory = mp.part.replay.RingBuffer(10, [[]])
        for number in range(5):
            memory.push(number)
        assert (memory[:][0] == [0, 1, 2, 3, 4]).all()
        assert (memory[1:4][0] == [1, 2, 3]).all()
        assert (memory[:-2][0] == [0, 1, 2]).all()
        assert (memory[-3:][0] == [2, 3, 4]).all()
        assert (memory[-2:-1][0] == [3]).all()

    def test_slicing_overfull(self):
        memory = mp.part.replay.RingBuffer(5, [[]])
        for number in range(10):
            memory.push(number)
        assert (memory[:][0] == [5, 6, 7, 8, 9]).all()
        assert (memory[5:8][0] == [5, 6, 7]).all()
        assert (memory[:-2][0] == [5, 6, 7]).all()
        assert (memory[-3:][0] == [7, 8, 9]).all()
        assert (memory[-2:-1][0] == [8]).all()

    def test_slicing_wrap_around(self):
        memory = mp.part.replay.RingBuffer(10, [[]])
        for number in range(12):
            memory.push(number)
        assert (memory[8:][0] == [8, 9, 10, 11]).all()
        assert (memory[-4:][0] == [8, 9, 10, 11]).all()


class TestSequential:

    def test_streaming_no_missing(self):
        memory = mp.part.replay.Sequential(1, [[]])
        for number in range(20):
            memory.push(number)
            assert memory.batch(1)[0] == number
        assert not len(memory)

    def test_override_oldest(self):
        memory = mp.part.replay.Sequential(5, [[]])
        for number in range(20):
            memory.push(number)
        assert (memory.batch(5)[0] == [15, 16, 17, 18, 19]).all()

    def test_free_after_batch(self):
        memory = mp.part.replay.Sequential(10, [[]])
        for number in range(20):
            memory.push(number)
        assert (memory.batch(5)[0] == [10, 11, 12, 13, 14]).all()
        assert (memory.batch(5)[0] == [15, 16, 17, 18, 19]).all()
        assert not len(memory)

    def test_shuffle(self):
        random = np.random.RandomState(0)
        memory = mp.part.replay.Sequential(10, [[]], random)
        for number in range(20):
            memory.push(number)
        memory.shuffle()
        batch = memory.batch(10)[0]
        assert (np.sort(batch) == list(range(10, 20))).all()
        assert not (batch == list(range(10, 20))).all()

    def test_shuffle_underfull(self):
        random = np.random.RandomState(0)
        memory = mp.part.replay.Sequential(20, [[]], random)
        for number in range(10):
            memory.push(number)
        memory.shuffle()
        batch = memory.batch(10)[0]
        assert (np.sort(batch) == list(range(10))).all()
        assert not (batch == list(range(10))).all()


class TestRandom:

    def test_streaming_no_missing(self):
        memory = mp.part.replay.Random(1, [[]])
        for number in range(20):
            memory.push(number)
            assert memory.batch(1)[0] == number
        assert len(memory) == 1

    def test_batch_shuffled(self):
        random = np.random.RandomState(0)
        memory = mp.part.replay.Random(10, [[]], random)
        for number in range(20):
            memory.push(number)
        batch = memory.batch(10)[0]
        assert (np.sort(batch) == list(range(10, 20))).all()
        assert not (batch == list(range(10, 20))).all()

    def test_batch_shuffled_replacement(self):
        random = np.random.RandomState(0)
        memory = mp.part.replay.Random(10, [[]], random, replace=True)
        for number in range(20):
            memory.push(number)
        batch = memory.batch(10)[0]
        assert all(10 <= x < 20 for x in batch)
        assert not (np.sort(batch) == list(range(10, 20))).all()

    def test_free_oldest_first(self):
        random = np.random.RandomState(0)
        memory = mp.part.replay.Random(10, [[]], random)
        for number in range(20):
            memory.push(number)
        batch = memory.batch(10)[0]
        assert all(10 <= x < 20 for x in batch)
