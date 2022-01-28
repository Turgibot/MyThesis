# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import ZedStreamer_pb2 as ZedStreamer__pb2


class ZedStreamerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SendImage = channel.unary_unary(
                '/zedstreamer.ZedStreamer/SendImage',
                request_serializer=ZedStreamer__pb2.Image.SerializeToString,
                response_deserializer=ZedStreamer__pb2.Received.FromString,
                )
        self.SendVideo = channel.stream_unary(
                '/zedstreamer.ZedStreamer/SendVideo',
                request_serializer=ZedStreamer__pb2.Image.SerializeToString,
                response_deserializer=ZedStreamer__pb2.Received.FromString,
                )
        self.SendParams = channel.unary_unary(
                '/zedstreamer.ZedStreamer/SendParams',
                request_serializer=ZedStreamer__pb2.Params.SerializeToString,
                response_deserializer=ZedStreamer__pb2.Received.FromString,
                )


class ZedStreamerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SendImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendVideo(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendParams(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ZedStreamerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SendImage': grpc.unary_unary_rpc_method_handler(
                    servicer.SendImage,
                    request_deserializer=ZedStreamer__pb2.Image.FromString,
                    response_serializer=ZedStreamer__pb2.Received.SerializeToString,
            ),
            'SendVideo': grpc.stream_unary_rpc_method_handler(
                    servicer.SendVideo,
                    request_deserializer=ZedStreamer__pb2.Image.FromString,
                    response_serializer=ZedStreamer__pb2.Received.SerializeToString,
            ),
            'SendParams': grpc.unary_unary_rpc_method_handler(
                    servicer.SendParams,
                    request_deserializer=ZedStreamer__pb2.Params.FromString,
                    response_serializer=ZedStreamer__pb2.Received.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'zedstreamer.ZedStreamer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ZedStreamer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SendImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/zedstreamer.ZedStreamer/SendImage',
            ZedStreamer__pb2.Image.SerializeToString,
            ZedStreamer__pb2.Received.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendVideo(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/zedstreamer.ZedStreamer/SendVideo',
            ZedStreamer__pb2.Image.SerializeToString,
            ZedStreamer__pb2.Received.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendParams(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/zedstreamer.ZedStreamer/SendParams',
            ZedStreamer__pb2.Params.SerializeToString,
            ZedStreamer__pb2.Received.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
