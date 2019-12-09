import binascii
import random
import datetime

import numpy as np
import matplotlib.pyplot as plt


class DatasetGenerator:
    def __init__(self,
                 training_source_path: str,
                 testing_source_path: str,
                 output_path: str):
        pass


class TraceConverter:
    def __init__(self,
                 input_file_list: list,
                 trace_bias=0.02,
                 amp_coef: int = 65535 * 20,
                 do_fft: bool = False,
                 target_byte: int = 0,
                 extend_times: int = 0,
                 rand_lowerbound: int = None,
                 rand_upperbound: int = None,
                 slice_start=None,
                 slice_end=None
                 ):
        self.__input_file_list = input_file_list
        self.__trace_bias = trace_bias
        self.__amp_coef = amp_coef
        self.__do_fft = do_fft
        self.__target_byte = target_byte

        self.__rand_shift = False
        self.__slice = True
        if slice_start is None and slice_end is None:
            self.__slice = False
        if self.__slice:
            assert slice_start is not None, "slice_start not set"
            assert slice_end is not None, "slice_end not set"
            self.__slice_start = slice_start
            self.__slice_end = slice_end

            if rand_lowerbound is not None and rand_upperbound is not None:
                self.__rand_shift = True
            if self.__rand_shift:
                assert rand_lowerbound is not None, "rand_lowerbound not set"
                assert rand_upperbound is not None, "rand_upperbound not set"
                self.__rnd_lowerbound = rand_lowerbound
                self.__rnd_upperbound = rand_upperbound

        self.__extend_times = extend_times

        self.__output_buff = list()
        self.__AES_Sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
        ])

    def convert(self):
        file_list = self.__input_file_list
        for file_path in file_list:
            self.__console_format("Processing: " + file_path)
            npydata = np.load(file_path, allow_pickle=True)
            key = npydata['argument'].item()['key']
            key_hex = np.array(list(binascii.unhexlify(key)))

            plaintext = npydata['argument'].item()['plaintext']
            plaintext_hex = np.array(list(binascii.unhexlify(plaintext)))

            ciphertext = npydata['argument'].item()['ciphertext'].replace(' ', '')
            ciphertext_hex = np.array(list(binascii.unhexlify(ciphertext)))

            label = self._labelize(plaintext_hex, key_hex)

            trace = npydata['channel'][0] + self.__trace_bias

            self.__output_buff.append(self._parse2queue(key, key_hex,
                                                        plaintext, plaintext_hex,
                                                        ciphertext, ciphertext_hex,
                                                        trace, label))
            if self.__extend_times > 0:
                for _ in range(self.__extend_times):
                    self.__output_buff.append(self._parse2queue(key, key_hex,
                                                                plaintext, plaintext_hex,
                                                                ciphertext, ciphertext_hex,
                                                                trace, label))
                # self.__debug_plot()
        # ===
        total_amount = len(self.__output_buff)
        key_len = len(self.__output_buff[0]['key_hex'])
        plaintext_len = len(self.__output_buff[0]['plaintext_hex'])
        ciphertext_len = len(self.__output_buff[0]['ciphertext_hex'])
        trace_len = len(self.__output_buff[0]['trace'])

        mkey = np.zeros((total_amount, key_len), dtype=np.uint8)
        mplaintext = np.zeros((total_amount, plaintext_len), dtype=np.uint8)
        mcihper = np.zeros((len(self.__output_buff), ciphertext_len), dtype=np.uint8)
        mtrace = np.zeros((total_amount, trace_len), dtype=np.float32)
        mlabel = np.zeros((total_amount), dtype=np.uint8)

        record_num = 0
        while self.__output_buff:
            data = self.__output_buff.pop()
            mkey[record_num] = data['key_hex']
            mplaintext[record_num] = data['plaintext_hex']
            mcihper[record_num] = data['ciphertext_hex']
            mtrace[record_num] = data['trace']
            mlabel[record_num] = data['label']

            record_num += 1

        # ======= Metadata ======
        metadata_desync = np.zeros(len(mtrace), np.uint32)
        metadata_type = np.dtype([("plaintext", mplaintext[0].dtype, (plaintext_len,)),
                                  ("ciphertext", mcihper[0].dtype, (ciphertext_len,)),
                                  ("key", mkey[0].dtype, (key_len,)),
                                  ("desync", np.uint32, (1,)),
                                  ])
        metadata = np.array(
            [(mplaintext[n], mcihper[n], mkey[n], metadata_desync[k])
             for n, k in
             zip(range(total_amount), range(len(metadata_desync)))],
            dtype=metadata_type)
        # =======================
        return {'key': mkey,
                'plaintext': mplaintext,
                'ciphertext': mcihper,
                'trace': mtrace,
                'label': mlabel,
                'metadata': metadata}

    def _parse2queue(self,
                     key, key_hex,
                     plaintext, plaintext_hex,
                     ciphertext, ciphertext_hex,
                     trace, label):
        return {'key': key,
                'key_hex': key_hex,
                'plaintext': plaintext,
                'plaintext_hex': plaintext_hex,
                'ciphertext': ciphertext,
                'ciphertext_hex': ciphertext_hex,
                'trace': self._process_trace(trace),
                'label': label
                }

    def _labelize(self, plaintext_hex, key_hex):
        return self.__AES_Sbox[plaintext_hex[self.__target_byte] ^ key_hex[self.__target_byte]]

    def __debug_plot(self):
        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_subplot(1, 1, 1)
        for data in self.__output_buff:
            trace = data['trace']
            ax.plot(range(len(trace)), trace)
        plt.show()
        fig.clf()
        plt.close()
        self.__output_buff.clear()

    def _process_trace(self, trace):
        if self.__rand_shift:
            assert self.__slice_end + self.__rnd_upperbound < len(trace), \
                "slice_end + rand_upperbound > len(trace) may cause problem."
            random_num = random.randint(self.__rnd_lowerbound, self.__rnd_upperbound)
            while self.__slice_end + random_num > len(trace):
                random_num = random.randint(self.__rnd_lowerbound, self.__rnd_upperbound)
            trace = trace[self.__slice_start + random_num: self.__slice_end + random_num]

        elif self.__slice:
            trace = trace[self.__slice_start: self.__slice_end]

        if self.__do_fft:
            trace = np.fft.fft(trace)
            trace = abs(trace)
        else:
            trace = trace * self.__amp_coef

        return trace

    def __console_format(self, message: str):
        print("[{0}]: {1}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message))
