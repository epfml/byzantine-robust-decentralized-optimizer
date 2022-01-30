import threading
import time
from queue import Queue

_MIN_SLEEP_INTERVAL = 1e-7


def _fill_queue(iterator, queue, device):
    """
    Fill the queue with data from iterator until either the queue is full
    or the iterator is empty.
    """
    # If the iterate
    while not queue.full():
        try:
            data, target = iterator.__next__()
            data = data.to(device)
            target = target.to(device)
            queue.put((data, target))
        except StopIteration:
            queue.put(None)
            break


def _async_queue_feeder(coord):
    while True:
        for iterator, queue in zip(coord.get_iterators(), coord.queues):
            _fill_queue(iterator, queue, coord.device)

        time.sleep(_MIN_SLEEP_INTERVAL)


class _QueueIterator(object):
    """
    An object to simulate `iter(dataloader)`.
    """

    def __init__(self, q, coord):
        self.q = q
        self.coord = coord

    def __next__(self):
        while self.q.empty():
            time.sleep(_MIN_SLEEP_INTERVAL)

        output = self.q.get()
        if output is None:
            self.coord.epoch_end()
        return output


class _AsynchronousDataLoader(object):
    """
    This is a dummy wrapper over the torch data loader.

    This data loader only ensures that user can:
    1) use iter(dataloader) to get data and labels; and
    2) raise StopIteration when **all iterators** stop.

    The 2) means an async dataloader alone cannot raise StopIteration
    but the coordinator should do it.
    """

    def __init__(self, dataloader, q, coord):
        self.q = q
        self.dataloader = dataloader
        self.coord = coord
        self.dataset = dataloader.dataset

        # TODO: improve
        if hasattr(dataloader, "sampler"):
            self.sampler = dataloader.sampler

    def __iter__(self):
        # Inform the coordinator that this queue has been launched.
        self.coord.epoch_start(self.q)

        # Every time this is called, signaling the coordinator to
        # Initialize the dataloader.
        return _QueueIterator(self.q, self.coord)


class AsyncDataLoaderCoordinator(object):
    """
    Example
    ```
    loader_coordinator = AsyncDataLoaderCoordinator(device=device)
    for rank in range(args.n):
        train_loader = task.train_loader(sampler=sampler)
        train_loader = loader_coordinator.add(train_loader)
    ```
    """

    # Note that the maxsize should not be too large that it runs out of data
    def __init__(self, device, maxsize: int = 20):
        self.maxsize = maxsize
        self.device = device

        self.queues = []
        self.dataloaders = []
        self.iterators = []

        # Not daemon thread
        self.t = threading.Thread(
            target=_async_queue_feeder, daemon=True, args=(self,))
        self.t.start()

        self.epoch_start_flag = set()

    def add(self, dataloader):
        queue = Queue(maxsize=self.maxsize)
        dummy_loader = _AsynchronousDataLoader(dataloader, queue, self)
        self.dataloaders.append(dummy_loader)
        self.queues.append(queue)
        return dummy_loader

    def get_iterators(self):
        return self.iterators

    def epoch_start(self, queue):
        self.iterators = []  # this stops the async process from adding None
        self.epoch_start_flag.add(queue)
        if len(self.epoch_start_flag) < len(self.queues):
            return

        def _clean_queues():
            # When self.iterators == [], `None` will not be added to the queue.
            assert self.iterators == [], len(self.iterators)
            for queue in self.queues:
                while not queue.empty():
                    output = queue.get()
                    # if output is not None:
                    #     raise NotImplementedError(
                    #         "The number of batches on each worker is not the same!")

        _clean_queues()
        self.epoch_start_flag = set()
        self.iterators = [iter(loader.dataloader)
                          for loader in self.dataloaders]

    def epoch_end(self):
        self.iterators = []  # this stops the async process from adding None
        raise StopIteration
