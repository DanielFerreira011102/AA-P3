import farmhash
import mmh3

import mmh3
import farmhash
import xxhash

class GeneralHashFunctions:
    """
    GeneralHashFunctions is a class containing several hash functions.
    """
    
    @staticmethod
    def murmurhash(key, b=32, seed=None, signed=True):
        """
        MurmurHash is a non-cryptographic hash function suitable for general hash-based lookup.
        
        :param key: The input data to be hashed.
        :param b: Bit size of the hash (32 or 64).
        :param seed: Seed value for hash function (optional, default is 0).
        :return: Hash value.
        """
        if b == 32:
            return mmh3.hash(key, seed, signed)
        elif b == 64:
            return mmh3.hash64(key, seed, signed)
        elif b == 128:
            return mmh3.hash128(key, seed, signed)

    @staticmethod
    def farmhash(key, b=32, seed=None):
        """
        FarmHash is a family of hash functions suitable for hash-based lookup and similar operations.
        
        :param key: The input data to be hashed.
        :param b: Bit size of the hash (32 or 64).
        :param seed: Seed value for hash function (optional, default is None).
        :return: Hash value.
        """
        if b == 32:
            return farmhash.hash32withseed(key, seed)
        elif b == 64:
            return farmhash.hash64withseed(key, seed)
        elif b == 128:
            return farmhash.hash128withseed(key, seed)
        
    @staticmethod
    def xxhash(key, b=32, seed=None):
        """
        xxHash is an extremely fast non-cryptographic hash algorithm, working at speeds close to RAM limits.
            
        :param key: The input data to be hashed.
        :param b: Bit size of the hash (32, 64 or 128).
        :param seed: Seed value for hash function (optional, default is 0).
        :return: Hash value.
        """
        if b == 32:
            return xxhash.xxh32(key, seed).intdigest()
        elif b == 64:
            return xxhash.xxh64(key, seed).intdigest()
        elif b == 128:
            return xxhash.xxh128(key, seed).intdigest()

    @staticmethod
    def djb2(key):
        """
        DJB2 is a simple hash function created by Daniel J. Bernstein.
        
        :param key: The input data to be hashed.
        :return: Hash value.
        """
        hash_value = 5381
        for char in key:
            hash_value = ((hash_value << 5) + hash_value) + ord(char)
        return hash_value & 0xFFFFFFFF

    
    @staticmethod
    def fnv1a(key, b=32):
        """
        FNV-1a is a non-cryptographic hash function that performs well for short keys.

        :param key: The input data to be hashed.
        :param b: Bit size of the hash (32, 64, 128, 256, 512, or 1024).
        :return: Hash value.
        """
        if b == 32:
            FNV_prime = 0x01000193
            FNV_offset_basis = 0x811c9dc5
        elif b == 64:
            FNV_prime = 0x00000100000001B3
            FNV_offset_basis = 0xcbf29ce484222325
        elif b == 128:
            FNV_prime = 0x0000000001000000000000000000013B	
            FNV_offset_basis = 0x6c62272e07bb014262b821756295c58d
        elif b == 256:
            FNV_prime = 0x0000000000000000000001000000000000000000000000000000000000000163
            FNV_offset_basis = 0xdd268dbcaac550362d98c384c4e576ccc8b1536847b6bbb31023b4c8caee0535
        elif b == 512:
            FNV_prime = 0x00000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000157
            FNV_offset_basis = 0xb86db0b1171f4416dca1e50f309990acac87d059c90000000000000000000d21e948f68a34c192f62ea79bc942dbe7ce182036415f56e34bac982aac4afe9fd9
        elif b == 1024:
            FNV_prime = 0x000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000018D
            FNV_offset_basis = 0x0000000000000000005f7a76758ecc4d32e56d5a591028b74b29fc4223fdada16c3bf34eda3674da9a21d9000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004c6d7eb6e73802734510a555f256cc005ae556bde8cc9c6a93b21aff4b16c71ee90b

        hash_value = FNV_offset_basis
        for char in key:
            hash_value = hash_value ^ ord(char)
            hash_value = hash_value * FNV_prime
        return hash_value & ((1 << b) - 1)

    @staticmethod
    def jenkins_one_at_a_time(key):
        """
        Jenkins One-at-a-Time is a non-cryptographic hash function that is simple and fast.

        :param key: The input data to be hashed.
        :return: Hash value.
        """
        hash_value = 0
        for char in key:
            hash_value += ord(char)
            hash_value += (hash_value << 10)
            hash_value ^= (hash_value >> 6)
        hash_value += (hash_value << 3)
        hash_value ^= (hash_value >> 11)
        hash_value += (hash_value << 15)
        return hash_value & 0xFFFFFFFF
    
    def single_char_hash(self, key):
        """
        SingleCharHash is a hash function that returns the ASCII value of the first character of the input string.

        :param key: The input data to be hashed.
        :return: Hash value.
        """
        return ord(key)
