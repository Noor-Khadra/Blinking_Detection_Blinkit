    '''the aim of this file is:
Acquire EEG samples from BITalino and deliver 
them to the rest of the application in batches. '''
import sys
print(sys.executable)
import logging
import numpy as np

log = logging.getLogger(__name__)


class BitalinoAcquisition:

    def __init__(
        self,
        mac_address,
        sampling_rate,
        channels,
        batch_samples
    ):
        self.mac_address = mac_address
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.batch_samples = batch_samples

        self.device = None
        self.started = False

    def connect(self):
        from bitalino import BITalino

        self.device = BITalino(self.mac_address)

        self.device.start(
            self.sampling_rate,
            self.channels
        )

        self.started = True

    def read_batch(self):
        raw = self.device.read(self.batch_samples)

        data = np.asarray(raw, dtype=np.float32)

        analog = data[:, 5:]

        return analog, None

    def stop(self):
        if self.device is not None and self.started:
            self.device.stop()
            self.started = False

    def close(self):
        if self.device is not None:
            self.device.close()
            self.device = None