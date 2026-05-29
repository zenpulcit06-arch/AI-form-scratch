module SaveingData
    use, intrinsic :: iso_fortran_env
    use Network
    implicit none

    private
    public save_network, load_network
contains
    subroutine save_network(net,filename)
        class(layer), intent(in) :: net(:)
        character(len=1000), intent(in) :: filename
        integer(int64) :: i

        open(unit=10, file=filename, form='unformatted', access='stream', status='new')
        write(10) int(size(net),int64)
        do i = 1, size(net)
            !b(current_neuron),W(prev_neuron,current_neuron)
           write(10) int(size(net(i)%b),int64), int(size(net(i)%W,1),int64), net(i)%W, net(i)%b
        end do
        close(10)

    end subroutine save_network

    subroutine load_network(net,filename,N)
        type(layer),allocatable, intent(out) :: net(:)
        character(len=1000), intent(in) :: filename
        integer(int64), intent(out) :: N
        integer(int64) :: i,prev_neuron,current_neuron

        open(unit=10, file=filename, form='unformatted', access='stream', status='old')
        read(10) N

        if (allocated(net)) deallocate(net)
        allocate(net(N))

        do i = 1, N
            read(10) current_neuron, prev_neuron
            call intialize_layer(net(i), prev_neuron, current_neuron, 1_int64)
            read(10) net(i)%W, net(i)%b
        end do

        close(10)
    end subroutine
end module SaveingData